const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("motionbloomBridge", {
  start: () => ipcRenderer.invoke("bridge:start"),
  stop: () => ipcRenderer.invoke("bridge:stop"),
  onEvent: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on("bridge:event", listener);
    return () => ipcRenderer.removeListener("bridge:event", listener);
  },
  openExternal: (url) => ipcRenderer.invoke("shell:openExternal", url),
  launchGame: (url, name) => ipcRenderer.invoke("game:launch", { url, name }),
  saveReport: (html, filename) => ipcRenderer.invoke("report:save", { html, filename }),
});
