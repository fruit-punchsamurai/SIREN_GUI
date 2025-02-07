// static/main.js
const clientId = uuidv4();
let peerConnection = null;
let dataChannel = null;
let currentFile = null;
let currentFileSize = 0;
let bytesSent = 0;
let targetId = null;
let receivingFile = null;
let bytesReceived = 0;

document.getElementById('clientId').textContent = clientId;

const ws = new WebSocket(`ws://${window.location.host}/ws/${clientId}`);

ws.onmessage = async (event) => {
    const message = JSON.parse(event.data);

    switch (message.event) {
        case 'assignIdentity':
            handleAssignIdentity(message.clientName, message.clientId);
            break;
        case 'updateClients':
            updateClientList(message.uniqueNames);
            break;
        case 'offer':
            await handleOffer(message);
            break;
        case 'answer':
            await handleAnswer(message);
            break;
        case 'ice-candidate':
            handleIceCandidate(message);
            break;
        case 'transferRequest':
            handleTransferRequest(message);
            break;
        case 'transferResponse':
            handleTransferResponse(message);
            break;
    }
};

function uuidv4() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        const r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

function handleAssignIdentity(clientName, clientId) {
    document.getElementById('clientName').textContent = clientName;
    document.getElementById('clientId').textContent = clientId;
    // client avatar id is img element change it's src to avatarSvg
    document.getElementById('clientAvatar').src = `avatars/${clientId}.svg`
}


function updateClientList(uniqueNames) {
    const clientList = document.getElementById('clientList');
    const noClients = document.getElementById('noClients');
    clientList.innerHTML = Object.keys(uniqueNames)
    .filter(id => id !== clientId)
    .map(id => {
        return `
        <div class="flex flex-col items-center text-center cursor-pointer hover:scale-[1.15] group transition-transform duration-300" onclick="initiateFileTransfer('${id}')">
            <img title="uuid: ${id}" src="avatars/${id}.svg" alt="${uniqueNames[id]}'s Avatar" class="w-16 h-16 rounded-full mb-2 border-2 border-white group-hover:border-green-500 shadow-lg" />
            <span title="uuid: ${id}" class="text-sm font-medium group-hover:shadow-xl group-hover:font-bold">${uniqueNames[id]}</span>
        </div>
    `;
    })
    .join('');

    if (Object.keys(uniqueNames).length == 1) {
        noClients.hidden = false;
    }
    else {
        noClients.hidden = true;
    }
}


document.getElementById('fileInput').addEventListener('change', handleFileSelect);
document.getElementById('clearFile').addEventListener('click', clearSelectedFile);

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        currentFile = file;
        currentFileSize = file.size;
        document.getElementById('select-button').textContent = "Change"
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('fileName-div').hidden = false
        document.getElementById('fileSize').textContent = formatFileSize(file.size);
        document.getElementById('clearFile').classList.remove('hidden');
    }
}

function clearSelectedFile() {
    currentFile = null;
    currentFileSize = 0;
    document.getElementById('fileInput').value = ""; // Clear file input
    document.getElementById('fileName').textContent = "";
    document.getElementById('fileName-div').hidden = true
    document.getElementById('fileSize').textContent = "";
    document.getElementById('clearFile').classList.add('hidden');
    document.getElementById('select-button').textContent = "Select File"
}

// function initiateFileTransfer(selectedTargetId) {
//     if (!currentFile) return alert('Please select a file first');
//     targetId = selectedTargetId;

//     ws.send(JSON.stringify({
//         event: 'transferRequest',
//         targetId: targetId,
//         fileName: currentFile.name,
//         fileSize: currentFile.size
//     }));
// }

function initiateFileTransfer(selectedTargetId) {
    if (!currentFile) return alert('Please select a file first');
    targetId = selectedTargetId;

    // Find the selected client element and add a "Waiting..." text below
    const selectedClient = document.querySelector(`[onclick="initiateFileTransfer('${targetId}')"]`);
    if (selectedClient) {
        let waitingText = selectedClient.querySelector('.waiting-text');
        if (!waitingText) {
            waitingText = document.createElement('span');
            waitingText.className = "waiting-text text-xs text-yellow-300 mt-1";
            waitingText.textContent = "Waiting...";
            selectedClient.appendChild(waitingText);
        }
    }

    ws.send(JSON.stringify({
        event: 'transferRequest',
        targetId: targetId,
        fileName: currentFile.name,
        fileSize: currentFile.size
    }));
}


async function handleTransferRequest(message) {
    // const accepted = confirm(`Accept file transfer of ${message.fileName} (${formatFileSize(message.fileSize)}) from ${message.senderId}?`);

    // Create a modal to confirm file transfer
    const modal = document.getElementById('modal');
    modal.classList.remove('hidden');
    document.getElementById('modalMessage').textContent = `Accept file transfer of ${message.fileName} (${formatFileSize(message.fileSize)}) from ${message.senderId}?`;

    const accepted = await new Promise((resolve) => {
        document.getElementById('acceptFile').onclick = () => resolve(true);
        document.getElementById('rejectFile').onclick = () => resolve(false);
    });

    if (accepted) {
        targetId = message.senderId;
        createPeerConnection(targetId);
    }
    modal.classList.add('hidden');

    ws.send(JSON.stringify({
        event: 'transferResponse',
        targetId: message.senderId,
        accepted: accepted
    }));
}

// function handleTransferResponse(message) {
//     if (message.accepted) {
//         createPeerConnection(targetId);
//         startSendingFile();
//     } else {
//         // alert('File transfer rejected');
//         document.querySelector(`[onclick="initiateFileTransfer('${message.senderId}')"] .waiting-text`) = 'Rejected';
//         resetTransfer();
//         clearSelectedFile();
//     }
// }

function handleTransferResponse(message) {
    const selectedClient = document.querySelector(`[onclick="initiateFileTransfer('${message.senderId}')"]`);
    
    if (selectedClient) {
        let waitingText = selectedClient.querySelector('.waiting-text');
        if (!waitingText) {
            waitingText = document.createElement('span');
            waitingText.className = "waiting-text text-xs mt-1";
            selectedClient.appendChild(waitingText);
        }

        if (message.accepted) {
            waitingText.textContent = "Accepted";
            waitingText.classList.remove('text-yellow-300', 'text-red-500');
            waitingText.classList.add('text-green-400');
            
            createPeerConnection(targetId);
            startSendingFile();
        } else {
            waitingText.textContent = "Rejected";
            waitingText.classList.remove('text-yellow-300', 'text-green-400');
            waitingText.classList.add('text-red-300');

            // Remove "Rejected" text after 3 seconds
            setTimeout(() => {
                waitingText.remove();
            }, 3000);

            resetTransfer();
            clearSelectedFile();
        }
    }
}


function createPeerConnection(targetId) {
    const config = {
        iceServers: [{ urls: "stun:stun.l.google.com:19302" }]
    };

    peerConnection = new RTCPeerConnection(config);

    peerConnection.onicecandidate = ({ candidate }) => {
        if (candidate) {
            ws.send(JSON.stringify({
                event: 'ice-candidate',
                targetId: targetId,
                candidate: candidate
            }));
        }
    };

    if (currentFile) { // We're the sender
        dataChannel = peerConnection.createDataChannel('fileTransfer');
        setupDataChannel();
    } else { // We're the receiver
        peerConnection.ondatachannel = (event) => {
            dataChannel = event.channel;
            setupDataChannel();
        };
    }
}

async function startSendingFile() {
    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);

    ws.send(JSON.stringify({
        event: 'offer',
        targetId: targetId,
        offer: offer
    }));
}

async function handleOffer(message) {
    await peerConnection.setRemoteDescription(new RTCSessionDescription(message.offer));
    const answer = await peerConnection.createAnswer();
    await peerConnection.setLocalDescription(answer);

    ws.send(JSON.stringify({
        event: 'answer',
        targetId: message.senderId,
        answer: answer
    }));
}

async function handleAnswer(message) {
    await peerConnection.setRemoteDescription(new RTCSessionDescription(message.answer));
}

function handleIceCandidate(message) {
    peerConnection.addIceCandidate(new RTCIceCandidate(message.candidate));
}

function setupDataChannel() {
    dataChannel.binaryType = 'arraybuffer';

    dataChannel.onopen = () => {
        if (currentFile) { // Sender
            // Initialize sender progress display
            document.getElementById('transferProgress').innerHTML = `
                <div class="w-1/2 text-center border-2 border-green-500 rounded-lg p-4">
                <h3 class="w-full flex flex-col"> Sending: <span class="w-full font-bold break-all">${currentFile.name}</span></h3>
                <progress class="w-full" value="0" max="${currentFile.size}"></progress>
                <div class="progress-text">0%</div>
                <button id="dismissProgressButton" onclick="handleDismissProgressButton()" class="bg-red-400 text-white rounded-lg shadow-lg hover:bg-red-500 transition duration-300 py-2 px-6" hidden>
                    Dismiss
                </button>
                </div>
            `;
            sendFileInChunks();
        }
    };

    dataChannel.onmessage = (event) => {
        if (!receivingFile) {
            // First message contains file metadata
            try {
                const fileInfo = JSON.parse(event.data);
                startReceivingFile(fileInfo);
            } catch (error) {
                console.error('Error parsing metadata:', error);
            }
        } else {
            // Subsequent messages contain file chunks
            if (event.data instanceof ArrayBuffer) {
                receiveFileChunk(event.data);
            } else {
                console.error('Received non-binary chunk:', event.data);
            }
        }
    };

    dataChannel.onclose = () => {
        if (receivingFile) {
            completeFileTransfer();
        }

        resetTransfer();
        clearSelectedFile();
        document.getElementById('dismissProgressButton').hidden = false;
    };
}

function handleDismissProgressButton() {
    document.getElementById('transferProgress').innerHTML = '';
}

function sendFileInChunks() {
    const chunkSize = 16384; // 16KB chunks
    const reader = new FileReader();
    let offset = 0;

    // First send file metadata
    dataChannel.send(JSON.stringify({
        fileName: currentFile.name,
        fileSize: currentFile.size
    }));

    reader.onload = (e) => {
        if (dataChannel.bufferedAmount > chunkSize * 2) {
            // Wait if buffer is full
            setTimeout(() => reader.readAsArrayBuffer(currentFile.slice(offset, offset + chunkSize)), 100);
        } else {
            dataChannel.send(e.target.result);
            offset += e.target.result.byteLength;
            updateProgress(offset, currentFileSize);

            if (offset < currentFileSize) {
                reader.readAsArrayBuffer(currentFile.slice(offset, offset + chunkSize));
            } else {
                dataChannel.close();
            }
        }
    };

    reader.readAsArrayBuffer(currentFile.slice(offset, offset + chunkSize));
}

function startReceivingFile(fileInfo) {
    receivingFile = {
        name: fileInfo.fileName,
        size: Number(fileInfo.fileSize) || 0, // Ensure numeric value
        data: []
    };

    // Initialize progress bar with proper max value
    document.getElementById('transferProgress').innerHTML = `
        <div class="w-1/2 text-center border-2 border-green-500 rounded-lg p-4">
        <h3 class="w-full flex flex-col">Receiving: <span class="w-full font-bold break-all">${receivingFile.name}</span></h3>
        <progress class="w-full" value="0" max="${receivingFile.size}"></progress>
        <div class="progress-text">0%</div>
        <button id="dismissProgressButton" onclick="handleDismissProgressButton()" class="bg-red-400 w-full text-white rounded-lg shadow-lg hover:bg-red-500 transition duration-300 py-2" hidden>
            Dismiss
        </button>
        </div>
    `;
}

function receiveFileChunk(chunk) {
    if (!(chunk instanceof ArrayBuffer)) {
        console.error('Invalid chunk type:', typeof chunk);
        return;
    }

    receivingFile.data.push(chunk);
    bytesReceived += chunk.byteLength;

    // Safeguard against NaN values
    const safeBytes = Number.isFinite(bytesReceived) ? bytesReceived : 0;
    const safeTotal = Number.isFinite(receivingFile.size) ? receivingFile.size : 1;

    updateProgress(safeBytes, safeTotal);
}

function completeFileTransfer() {
    const receivedFile = new Blob(receivingFile.data);
    const url = URL.createObjectURL(receivedFile);
    const a = document.createElement('a');
    a.href = url;
    a.download = receivingFile.name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    // alert(`File ${receivingFile.name} received successfully!`);
    // document.getElementById('dismissProgressButton').hidden = false;

    resetTransfer();
}

function updateProgress(bytes, total) {
    const progress = document.querySelector('#transferProgress div progress');
    const progressText = document.querySelector('#transferProgress div .progress-text');

    if (progress && progressText) {
        // Ensure valid numeric values
        const validBytes = Number.isFinite(bytes) ? bytes : 0;
        const validTotal = Number.isFinite(total) && total > 0 ? total : 1;

        progress.value = validBytes;
        progress.max = validTotal;
        const percent = (validBytes / validTotal) * 100;
        progressText.textContent = `${percent.toFixed(2)}%`;
    }
}

function resetTransfer() {
    currentFile = null;
    currentFileSize = 0;
    bytesSent = 0;
    bytesReceived = 0;
    receivingFile = null;
    targetId = null;
    if (dataChannel) dataChannel.close();
    if (peerConnection) peerConnection.close();
    dataChannel = null;
    peerConnection = null;
}

function formatFileSize(bytes) {
    if (bytes >= 1073741824) return (bytes / 1073741824).toFixed(2) + ' GB';
    if (bytes >= 1048576) return (bytes / 1048576).toFixed(2) + ' MB';
    if (bytes >= 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return bytes + ' bytes';
}