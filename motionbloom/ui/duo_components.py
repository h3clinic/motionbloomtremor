"""Duolingo-style UI components with 3D depth, gamification, and playful interactions.

Provides reusable card classes that embody the Duolingo aesthetic:
- 3D "stamped" appearance with thick borders
- Vibrant color palette (green for success, blue for info, orange for streak)
- Hover effects for tactile feedback
- Mascot-centric hero card layout
"""

try:
    import customtkinter as ctk
    from tkinter import Label as TkLabel
    HAS_CTK = True
except ImportError:
    HAS_CTK = False

from . import theme


class DuoCard(ctk.CTkFrame if HAS_CTK else object):
    """3D-styled Duolingo card with rounded corners and thick border for depth.
    
    Features:
    - Dark background with high-contrast borders
    - Rounded corners (16px) for modern look
    - Hover effects: border color transitions
    - Internal padding and grid layout ready
    """

    def __init__(self, master, hover_enabled=True, **kwargs):
        """Initialize DuoCard.
        
        Args:
            master: Parent widget
            hover_enabled: Whether to bind hover effects (default True)
            **kwargs: Additional CTkFrame arguments
        """
        if not HAS_CTK:
            raise ImportError("CustomTkinter required for DuoCard")
        
        super().__init__(
            master,
            fg_color=theme.SURFACE,
            corner_radius=theme.CTK_FRAME_CORNER_RADIUS,
            border_width=2,
            border_color=theme.BORDER,
            **kwargs
        )
        
        # Internal grid for flexible layouts
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        if hover_enabled:
            self.bind_hover()
    
    def bind_hover(self):
        """Add hover effect: border color transitions on mouse enter/leave."""
        def on_enter(event):
            self.configure(border_color=theme.PRIMARY)
        
        def on_leave(event):
            self.configure(border_color=theme.BORDER)
        
        self.bind("<Enter>", on_enter)
        self.bind("<Leave>", on_leave)


class DuoHeroCard(DuoCard):
    """Hero card with mascot (left), motivational text (center), and stats (right).
    
    Layout:
    - Column 0: 80×80 mascot frame (fixed)
    - Column 1: Title + subtitle (flexible, center-weighted)
    - Column 2: Stats/streak badge (fixed, right-aligned)
    
    Design embodies Duolingo's gamification:
    - Mascot reacts to user state (idle → analyzing → success)
    - Bold, encouraging text ("You're on fire!", "Keep it up!")
    - Prominent streak counter in warm orange
    """

    def __init__(self, master, mascot_emoji="🎯", **kwargs):
        """Initialize DuoHeroCard.
        
        Args:
            master: Parent widget
            mascot_emoji: Placeholder emoji before AI asset integration
            **kwargs: Additional DuoCard arguments
        """
        super().__init__(master, **kwargs)
        
        # Layout: 3 columns (mascot | text | stats)
        self.grid_columnconfigure(0, weight=0)  # Mascot (fixed width)
        self.grid_columnconfigure(1, weight=1)  # Text (flexible)
        self.grid_columnconfigure(2, weight=0)  # Stats (fixed width)
        
        # --- Column 0: Mascot Frame (80×80) ---
        self.mascot_frame = ctk.CTkFrame(
            self,
            fg_color=theme.SURFACE_ALT,
            width=80,
            height=80,
            corner_radius=12
        )
        self.mascot_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nw")
        self.mascot_frame.grid_propagate(False)
        
        # Mascot emoji/image placeholder
        self.mascot_label = ctk.CTkLabel(
            self.mascot_frame,
            text=mascot_emoji,
            font=("Arial", 48),
            text_color=theme.TEXT
        )
        self.mascot_label.pack(fill="both", expand=True)
        
        # --- Column 1: Text Container (Motivational) ---
        self.text_container = ctk.CTkFrame(self, fg_color="transparent")
        self.text_container.grid(row=0, column=1, padx=16, pady=20, sticky="ew")
        
        # Title
        self.title_label = ctk.CTkLabel(
            self.text_container,
            text="Welcome back!",
            font=("Helvetica", 20, "bold"),
            text_color=theme.TEXT
        )
        self.title_label.pack(anchor="w", pady=(0, 4))
        
        # Subtitle
        self.subtitle_label = ctk.CTkLabel(
            self.text_container,
            text="Ready to check your movement?",
            font=("Helvetica", 14),
            text_color=theme.MUTED,
            wraplength=300,
            justify="left"
        )
        self.subtitle_label.pack(anchor="w")
        
        # --- Column 2: Stats Container (Right) ---
        self.stats_container = ctk.CTkFrame(self, fg_color="transparent")
        self.stats_container.grid(row=0, column=2, padx=20, pady=20, sticky="ne")
        
        # Streak badge (motivational orange)
        self.streak_label = ctk.CTkLabel(
            self.stats_container,
            text="🔥 7",
            font=("Helvetica", 18, "bold"),
            text_color=theme.WARMTH
        )
        self.streak_label.pack(anchor="e")
        
        # Streak label
        self.streak_text_label = ctk.CTkLabel(
            self.stats_container,
            text="day streak",
            font=("Helvetica", 11),
            text_color=theme.MUTED
        )
        self.streak_text_label.pack(anchor="e", pady=(2, 0))
    
    def set_mascot_emoji(self, emoji: str):
        """Update mascot emoji/character."""
        self.mascot_label.configure(text=emoji)
    
    def set_title(self, text: str):
        """Update hero card title."""
        self.title_label.configure(text=text)
    
    def set_subtitle(self, text: str):
        """Update hero card subtitle/motivational text."""
        self.subtitle_label.configure(text=text)
    
    def set_streak(self, count: int):
        """Update streak counter."""
        self.streak_label.configure(text=f"🔥 {count}")
    
    def set_streak_hidden(self, hidden: bool = True):
        """Hide/show the streak badge."""
        if hidden:
            self.stats_container.grid_remove()
        else:
            self.stats_container.grid()


class DuoMetricsCard(DuoCard):
    """Metrics display card with 2-column grid of diagnostic labels.
    
    Layout: Each metric is a label pair (name, value) arranged in columns.
    Used for tremor metrics, quality scores, etc.
    """

    def __init__(self, master, title="Metrics", **kwargs):
        """Initialize DuoMetricsCard.
        
        Args:
            master: Parent widget
            title: Card title
            **kwargs: Additional DuoCard arguments
        """
        super().__init__(master, **kwargs)
        
        # Title
        self.title_label = ctk.CTkLabel(
            self,
            text=title,
            font=("Helvetica", 16, "bold"),
            text_color=theme.TEXT
        )
        self.title_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(16, 12))
        
        # Metrics grid (will be populated dynamically)
        self.metrics_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.metrics_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 16))
        self.metrics_frame.grid_columnconfigure(0, weight=1)
        self.metrics_frame.grid_columnconfigure(1, weight=1)
        
        self.metric_labels = {}
        self.metric_row = 0
    
    def add_metric(self, key: str, label: str, value: str = "-"):
        """Add a metric row (label, value pair).
        
        Args:
            key: Unique identifier for this metric
            label: Display label (e.g., "Hand Stillness")
            value: Initial value (e.g., "Steady")
        """
        row = self.metric_row
        col = row % 2
        row_idx = row // 2
        
        # Label
        lbl = ctk.CTkLabel(
            self.metrics_frame,
            text=label,
            font=("Helvetica", 12),
            text_color=theme.MUTED,
            anchor="w"
        )
        lbl.grid(row=row_idx, column=col * 2, sticky="w", padx=(0, 10), pady=4)
        
        # Value
        val = ctk.CTkLabel(
            self.metrics_frame,
            text=value,
            font=("Helvetica", 13, "bold"),
            text_color=theme.TEXT,
            anchor="e"
        )
        val.grid(row=row_idx, column=col * 2 + 1, sticky="e", pady=4)
        
        self.metric_labels[key] = val
        self.metric_row += 1
    
    def update_metric(self, key: str, value: str):
        """Update metric value by key."""
        if key in self.metric_labels:
            self.metric_labels[key].configure(text=value)


class DuoVerdictCard(DuoCard):
    """Verdict/status card with color-coded badge and message.
    
    Used to display: "Reading in progress", "All clear!", "Hand moving", etc.
    Badge color indicates status (green = good, red = warning, blue = info).
    """

    def __init__(self, master, status_color=theme.PRIMARY, message="", **kwargs):
        """Initialize DuoVerdictCard.
        
        Args:
            master: Parent widget
            status_color: Badge color (theme.PRIMARY, theme.RED, etc.)
            message: Status message text
            **kwargs: Additional DuoCard arguments
        """
        super().__init__(master, **kwargs)
        
        self.status_color = status_color
        
        # Status badge (colored dot + text)
        self.status_frame = ctk.CTkFrame(self, fg_color=status_color, corner_radius=8)
        self.status_frame.grid(row=0, column=0, padx=16, pady=16, sticky="ew")
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text=message,
            font=("Helvetica", 14, "bold"),
            text_color="#FFFFFF",
            padx=12,
            pady=10,
            anchor="w"
        )
        self.status_label.pack(fill="both", expand=True)
    
    def set_status(self, message: str, color: str = None):
        """Update status message and color.
        
        Args:
            message: New status message
            color: New badge color (if None, keeps current color)
        """
        if color:
            self.status_color = color
            self.status_frame.configure(fg_color=color)
        self.status_label.configure(text=message)
