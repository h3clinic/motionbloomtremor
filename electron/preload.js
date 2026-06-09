const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("motionbloomBridge", {
  start: () => ipcRenderer.invoke("bridge:start"),
  stop: () => ipcRenderer.invoke("bridge:stop"),
  onEvent: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on("bridge:event", listener);
    return () => ipcRenderer.removeListener("bridge:event", listener);
  },
});
