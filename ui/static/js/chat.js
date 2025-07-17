// Main chat interface functionality

// DOM element references
const ttmForm = get(".ttm-inputarea");
const ttmInput = get(".ttm-input");
const ttmChat = get(".ttm-chat");

// User input history for arrow key navigation
let userInputs = [];
let userInputIndex = 0;

// Handle form submission (user sends a message)
ttmForm.addEventListener("submit", event => {
    event.preventDefault();
    const msgText = ttmInput.value;
    if (!msgText) return;
    
    // Add user message to chat
    addToChat("right", msgText, "");
    
    // Show loading indicator and scroll
    insertLoadingDots();
    ttmChat.scrollTop += 200;
    
    // Clear input and send to bot
    ttmInput.value = "";
    botResponse(msgText);
});

// Show typing indicator while waiting for bot response
function insertLoadingDots() {
    let msgHTML = `
        <div class="loading-dots" id="current-dots">
            <div class="msg left-msg">
                <div class="msg-bubble">
                    <div class="stage">
                        <div class="dot-pulse"></div>
                    </div>
                </div>
            </div>
        </div>
    `
    ttmChat.insertAdjacentHTML("beforeend", msgHTML);
}

// Remove typing indicator when bot responds
function deleteLoadingDots() {
    document.getElementById("current-dots").outerHTML = "";
}

/**
 * Add a message to the chat interface
 * @param {string} side - "left" for bot messages, "right" for user messages
 * @param {string} text - Message content
 * @param {string} logText - Message ID for feedback logging
 */
function addToChat(side, text, logText) {
    let msgHTML = "";

    if (side == "left") {
        // Bot message - clean format without feedback
        msgHTML = `
            <div class="msg ${side}-msg">
                <div class="msg-bubble">
                    <div class="msg-text">${text}</div>
                </div>
            </div>
        `;
        ttmChat.insertAdjacentHTML("beforeend", msgHTML);
    } else {
        // User message - simpler format, add to history
        userInputIndex = userInputs.push(text);
        msgHTML = `
            <div class="msg ${side}-msg">
                <div class="msg-bubble">
                    <div class="msg-text">${text}</div>
                </div>
            </div>
        `;
        ttmChat.insertAdjacentHTML("beforeend", msgHTML);
    }
    // Auto-scroll to bottom
    ttmChat.scrollTop += 500;
}

/**
 * Send user message to backend and handle response
 * @param {string} rawText - User's input text
 */
function botResponse(rawText) {
    let dataPackage = {userInput: rawText, userName: currentUserId}; // currentUserId will be set from template
    
    const result = $.ajax({
        type: 'POST',
        url: botResponseUrl, // Will be set from template
        data: JSON.stringify(dataPackage),
        contentType: 'application/json',
        cache: false,
        success: function (data) {
            // Response format: "message<>logId"
            const splitData = data.split('<>')
            const msgText = splitData[0];
            const logText = splitData[1];
            deleteLoadingDots();
            addToChat("left", msgText, logText);
        },
        error: function(xhr, status, error) {
            console.error('AJAX Error:', error);
            deleteLoadingDots();
            addToChat("left", "Sorry, I encountered an error. Please try again.", "Error");
        }
    });
}

// Simple querySelector wrapper
function get(selector, root = document) {
    return root.querySelector(selector);
}