document.addEventListener('DOMContentLoaded', function () {
    const term = new Terminal({
        cursorBlink: true,
        theme: {
            background: '#000000',
            foreground: '#ffffff'
        }
    });
    const fitAddon = new FitAddon.FitAddon();
    term.loadAddon(fitAddon);
    term.open(document.getElementById('terminal'));
    fitAddon.fit();

    let currentLine = '';
    let commandHistory = [];
    let historyIndex = -1;

    const prompt = () => {
        term.write('\r\n$ ');
    };

    term.writeln('Welcome to the Terminal Chatbot!');
    term.writeln('Type "help" for a list of commands.');
    prompt();

    term.onKey(({ key, domEvent }) => {
        const printable = !domEvent.altKey && !domEvent.ctrlKey && !domEvent.metaKey;

        if (domEvent.keyCode === 13) { // Enter
            if (currentLine.trim() !== '') {
                commandHistory.push(currentLine);
                historyIndex = commandHistory.length;
                handleCommand(currentLine);
            }
            currentLine = '';
            prompt();
        } else if (domEvent.keyCode === 8) { // Backspace
            if (currentLine.length > 0) {
                term.write('\b \b');
                currentLine = currentLine.slice(0, -1);
            }
        } else if (domEvent.keyCode === 38) { // Up arrow
            if (historyIndex > 0) {
                historyIndex--;
                term.write('\x1b[2K\r$ ' + commandHistory[historyIndex]);
                currentLine = commandHistory[historyIndex];
            }
        } else if (domEvent.keyCode === 40) { // Down arrow
            if (historyIndex < commandHistory.length - 1) {
                historyIndex++;
                term.write('\x1b[2K\r$ ' + commandHistory[historyIndex]);
                currentLine = commandHistory[historyIndex];
            } else {
                historyIndex = commandHistory.length;
                term.write('\x1b[2K\r$ ');
                currentLine = '';
            }
        } else if (printable) {
            currentLine += key;
            term.write(key);
        }
    });

    function handleCommand(command) {
        const [cmd, ...args] = command.trim().split(' ');
        term.writeln(''); // new line after command

        switch (cmd) {
            case 'help':
                term.writeln('Available commands:');
                term.writeln('  chat <query>         - Chat with the AI');
                term.writeln('  mode <rag|serp|none> - Set the chat mode');
                term.writeln('  rag_mode <pdf|txt|url|github> - Set the RAG mode');
                term.writeln('  upload               - Upload a file for RAG');
                term.writeln('  clear                - Clear the terminal');
                term.writeln('  help                 - Show this help message');
                break;
            case 'clear':
                term.clear();
                break;
            case 'chat':
                if (args.length > 0) {
                    const query = args.join(' ');
                    fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: query })
                    })
                    .then(response => response.json())
                    .then(data => {
                        term.writeln(data.response);
                    })
                    .catch(error => {
                        term.writeln(`Error: ${error}`);
                    });
                } else {
                    term.writeln('Usage: chat <query>');
                }
                break;
            case 'mode':
                if (args.length === 1 && ['rag', 'serp', 'none'].includes(args[0])) {
                    const use_serp = args[0] === 'serp';
                    const rag_mode = args[0] === 'rag' ? 'pdf' : null; // default to pdf if rag is selected
                    fetch('/set_mode', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ use_serp: use_serp, rag_mode: rag_mode })
                    })
                    .then(() => term.writeln(`Mode set to ${args[0]}`))
                    .catch(error => term.writeln(`Error: ${error}`));
                } else {
                    term.writeln('Usage: mode <rag|serp|none>');
                }
                break;
            case 'rag_mode':
                if (args.length === 1 && ['pdf', 'txt', 'url', 'github'].includes(args[0])) {
                    fetch('/set_mode', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ rag_mode: args[0] })
                    })
                    .then(() => term.writeln(`RAG mode set to ${args[0]}`))
                    .catch(error => term.writeln(`Error: ${error}`));
                } else {
                    term.writeln('Usage: rag_mode <pdf|txt|url|github>');
                }
                break;
            case 'upload':
                document.getElementById('file-input').click();
                break;
            default:
                term.writeln(`Unknown command: ${cmd}`);
        }
    }

    document.getElementById('file-input').addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            const formData = new FormData();
            formData.append('file', file);
            term.writeln(`Uploading ${file.name}...`);
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                term.writeln(data.message);
                prompt();
            })
            .catch(error => {
                term.writeln(`Error: ${error}`);
                prompt();
            });
        }
    });

    window.addEventListener('resize', () => {
        fitAddon.fit();
    });
});
