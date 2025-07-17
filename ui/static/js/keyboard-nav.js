// Keyboard navigation for input history

// Arrow key navigation through input history
document.onkeydown = checkKey;

function checkKey(e) {
    e = e || window.event;

    if (e.keyCode == '38') {
        // Up arrow - go to previous input
        if (userInputs.length == 0) return;
        
        userInputIndex = userInputIndex - 1;
        if (userInputIndex < 0) {
            userInputIndex = 0;
        }
        ttmInput.value = userInputs[userInputIndex];
    } else if (e.keyCode == '40') {
        // Down arrow - go to next input
        if (userInputs.length == 0) return;
        
        userInputIndex = userInputIndex + 1;
        if (userInputIndex > userInputs.length - 1) {
            userInputIndex = userInputs.length - 1;
        }
        ttmInput.value = userInputs[userInputIndex];
    }
}