function currentTime() {

    return new Date().toLocaleTimeString();

}

function terminalLog(message, type = "info") {

    const terminal = document.getElementById("terminal");

    const line = document.createElement("div");

    line.className = "terminal-line";

    line.innerHTML = `
        <span class="time">${currentTime()}</span>
        <span class="${type}">${message}</span>
    `;

    terminal.appendChild(line);

    terminal.scrollTop = terminal.scrollHeight;

}

window.onload = () => {

    terminalLog("System Ready", "success");

};
