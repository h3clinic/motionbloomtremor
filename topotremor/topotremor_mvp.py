import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, welch
from ripser import ripser


VIDEO_PATH = "videos/tremor.mp4"
OUTPUT_PREFIX = "outputs/tremor"


def bandpass(signal, fs, low=3.0, high=12.0, order=3):
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq

    if high_norm >= 1:
        high_norm = 0.99

    b, a = butter(order, [low_norm, high_norm], btype="band")
    return filtfilt(b, a, signal)


def time_delay_embedding(signal, delay=3, dimension=3):
    """
    Converts a 1D motion signal into a point cloud:
    [x(t), x(t+delay), x(t+2delay)]
    """
    n = len(signal) - delay * (dimension - 1)
    if n <= 0:
        raise ValueError("Signal too short for embedding.")

    embedded = np.zeros((n, dimension))

    for i in range(dimension):
        embedded[:, i] = signal[i * delay : i * delay + n]

    return embedded


def longest_h1_lifetime(diagram):
    """
    H1 = loops.
    A clean cyclic tremor should create stronger H1 persistence.
    """
    if diagram is None or len(diagram) == 0:
        return 0.0

    lifetimes = []

    for birth, death in diagram:
        if np.isfinite(death):
            lifetimes.append(death - birth)

    if not lifetimes:
        return 0.0

    return float(np.max(lifetimes))


def select_roi_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()

    if not ok:
        raise RuntimeError("Could not read video.")

    roi = cv2.selectROI("Select hand ROI, then press ENTER", frame, False, False)
    cv2.destroyWindow("Select hand ROI, then press ENTER")

    cap.release()

    x, y, w, h = roi
    if w == 0 or h == 0:
        raise RuntimeError("No ROI selected.")

    return x, y, w, h


def extract_optical_flow_signal(video_path, roi):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    x, y, w, h = roi

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame.")

    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(first_gray)
    mask[y : y + h, x : x + w] = 255

    points = cv2.goodFeaturesToTrack(
        first_gray,
        mask=mask,
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )

    if points is None or len(points) < 10:
        raise RuntimeError("Not enough trackable points in ROI.")

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    prev_gray = first_gray
    prev_points = points

    dx_series = []
    dy_series = []
    tracked_counts = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_points, None, **lk_params
        )

        if next_points is None or status is None:
            break

        good_new = next_points[status.flatten() == 1].reshape(-1, 2)
        good_old = prev_points[status.flatten() == 1].reshape(-1, 2)

        if len(good_new) < 10:
            break

        movement = good_new - good_old

        # Median is more robust than mean against bad points.
        dx = np.median(movement[:, 0])
        dy = np.median(movement[:, 1])

        dx_series.append(dx)
        dy_series.append(dy)
        tracked_counts.append(len(good_new))

        prev_gray = gray
        prev_points = good_new.reshape(-1, 1, 2)

    cap.release()

    dx_series = np.array(dx_series)
    dy_series = np.array(dy_series)
    tracked_counts = np.array(tracked_counts)

    if len(dx_series) < fps * 3:
        raise RuntimeError("Video too short or tracking failed too early.")

    return dx_series, dy_series, tracked_counts, fps


def analyze_signal(signal, fs):
    # Remove DC drift.
    signal = signal - np.mean(signal)

    # Tremor-band isolate.
    filtered = bandpass(signal, fs, low=3.0, high=12.0)

    # FFT / PSD feature.
    freqs, power = welch(filtered, fs=fs, nperseg=min(256, len(filtered)))

    tremor_mask = (freqs >= 3.0) & (freqs <= 12.0)
    tremor_freqs = freqs[tremor_mask]
    tremor_power = power[tremor_mask]

    peak_frequency = float(tremor_freqs[np.argmax(tremor_power)])
    peak_power = float(np.max(tremor_power))

    total_power = float(np.sum(power) + 1e-9)
    tremor_band_power = float(np.sum(tremor_power))
    tremor_power_ratio = tremor_band_power / total_power

    # Topological feature.
    # Takens delay: choose ~a quarter of the dominant period so the delay
    # embedding of a rhythmic signal traces an OPEN loop (phase advance ≈ 90°
    # per step). A FIXED delay (e.g. 3 frames) aliases 4–12 Hz tremor at 30 fps
    # (216°/step at 6 Hz) and COLLAPSES the loop — which makes cyclic tremor
    # score LOWER H1 than noise. Deriving the delay from peak_frequency fixes
    # the inversion and is what makes the milestone pass.
    if peak_frequency > 0:
        delay = int(max(1, round(fs / (4.0 * peak_frequency))))
    else:
        delay = 1

    embedded = time_delay_embedding(filtered, delay=delay, dimension=3)

    # Uniform (scalar) scaling normalises amplitude WITHOUT distorting the loop.
    # Per-axis z-scoring is avoided on purpose: it stretches low-variance axes
    # and lets broadband noise fake topological holes.
    embedded = (embedded - embedded.mean(axis=0)) / (embedded.std() + 1e-9)

    result = ripser(embedded, maxdim=1)
    diagrams = result["dgms"]

    h1_diagram = diagrams[1]
    h1_lifetime = longest_h1_lifetime(h1_diagram)

    return {
        "filtered": filtered,
        "freqs": freqs,
        "power": power,
        "embedded": embedded,
        "diagrams": diagrams,
        "peak_frequency": peak_frequency,
        "peak_power": peak_power,
        "tremor_power_ratio": tremor_power_ratio,
        "embedding_delay": delay,
        "h1_lifetime": h1_lifetime,
    }


def save_plots(raw_signal, analysis, tracked_counts, fs, prefix):
    t = np.arange(len(raw_signal)) / fs

    plt.figure()
    plt.plot(t, raw_signal)
    plt.xlabel("Time (s)")
    plt.ylabel("Raw median motion")
    plt.title("Raw Optical Flow Motion Signal")
    plt.savefig(f"{prefix}_raw_signal.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(t, analysis["filtered"])
    plt.xlabel("Time (s)")
    plt.ylabel("Filtered motion")
    plt.title("Bandpassed Tremor Signal: 3-12 Hz")
    plt.savefig(f"{prefix}_filtered_signal.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.semilogy(analysis["freqs"], analysis["power"])
    plt.axvline(analysis["peak_frequency"], linestyle="--")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title("Power Spectral Density")
    plt.savefig(f"{prefix}_psd.png", dpi=200, bbox_inches="tight")
    plt.close()

    embedded = analysis["embedded"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(embedded[:, 0], embedded[:, 1], embedded[:, 2], s=4)
    ax.set_title("Time-Delay Embedding Point Cloud")
    ax.set_xlabel("x(t)")
    ax.set_ylabel("x(t + τ)")
    ax.set_zlabel("x(t + 2τ)")
    plt.savefig(f"{prefix}_embedding.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    h1 = analysis["diagrams"][1]
    if len(h1) > 0:
        births = h1[:, 0]
        deaths = h1[:, 1]
        finite = np.isfinite(deaths)
        plt.scatter(births[finite], deaths[finite])
        max_val = max(np.max(births[finite]), np.max(deaths[finite])) if np.any(finite) else 1
        plt.plot([0, max_val], [0, max_val], linestyle="--")
    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.title("H1 Persistence Diagram: Loops")
    plt.savefig(f"{prefix}_h1_persistence.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(tracked_counts)
    plt.xlabel("Frame")
    plt.ylabel("Tracked points")
    plt.title("Tracking Quality")
    plt.savefig(f"{prefix}_tracked_points.png", dpi=200, bbox_inches="tight")
    plt.close()


def main():
    print("Selecting hand ROI...")
    roi = select_roi_first_frame(VIDEO_PATH)

    print("Extracting optical flow...")
    dx, dy, tracked_counts, fs = extract_optical_flow_signal(VIDEO_PATH, roi)

    # Use combined motion magnitude, but preserve sign from x for rhythm.
    # Later we should analyze x and y separately.
    signal = dx

    print("Analyzing signal...")
    analysis = analyze_signal(signal, fs)

    print("Saving plots...")
    save_plots(signal, analysis, tracked_counts, fs, OUTPUT_PREFIX)

    print("\n=== TopoTremor MVP Results ===")
    print(f"FPS: {fs:.2f}")
    print(f"Frames analyzed: {len(signal)}")
    print(f"Peak tremor frequency: {analysis['peak_frequency']:.2f} Hz")
    print(f"Tremor power ratio: {analysis['tremor_power_ratio']:.4f}")
    print(f"Embedding delay (auto): {analysis['embedding_delay']} frames")
    print(f"H1 longest loop lifetime: {analysis['h1_lifetime']:.4f}")
    print(f"Median tracked points: {np.median(tracked_counts):.0f}")

    print("\nSaved outputs:")
    print(f"{OUTPUT_PREFIX}_raw_signal.png")
    print(f"{OUTPUT_PREFIX}_filtered_signal.png")
    print(f"{OUTPUT_PREFIX}_psd.png")
    print(f"{OUTPUT_PREFIX}_embedding.png")
    print(f"{OUTPUT_PREFIX}_h1_persistence.png")
    print(f"{OUTPUT_PREFIX}_tracked_points.png")


if __name__ == "__main__":
    main()
