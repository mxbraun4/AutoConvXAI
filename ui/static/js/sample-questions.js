// Sample question generation system

/**
 * Requests a sample question from the server for a given action type
 * @param {string} rawAction - The action type to generate a sample for
 */
function doSample(rawAction) {
    let username = currentUserId; // Will be set from template
    let dataPackage = {action: rawAction, thisUserName: username}

    const result = $.ajax({
        type: 'POST',
        url: samplePromptUrl, // Will be set from template
        data: JSON.stringify(dataPackage),
        contentType: 'application/json',
        cache: false,
        success: function (data) {
            // Handle different response formats from the server
            if (typeof data === 'string') {
                // Simple string response
                ttmInput.value = data;
            } else if (data.samples && data.samples.length > 0) {
                // Array of samples - pick one randomly
                ttmInput.value = data.samples[Math.floor(Math.random() * data.samples.length)];
            } else {
                // Fallback if no valid samples returned
                ttmInput.value = "Ask me about the diabetes dataset!";
            }
        },
        error: function(xhr, status, error) {
            console.error('Sample prompt error:', error);
            ttmInput.value = "Ask me about the diabetes dataset!";
        }
    });
}

// Sample question functions - each maps to a specific action type
// Data Exploration
function sampleData() { doSample("data"); }
function sampleStatistic() { doSample("statistic"); }
function sampleLabels() { doSample("label"); }
function sampleDefine() { doSample("define"); }

// Filtering
function sampleFilter() { doSample("filter"); }

// Model Analysis
function sampleNewPred() { doSample("new_pred"); }
function sampleWhy() { doSample("explain"); }
function sampleImportant() { doSample("important"); }
function samplePerformance() { doSample("score"); }
function sampleModel() { doSample("model"); }

// What-If Analysis
function sampleWhatIf() { doSample("whatif"); }
function sampleFlip() { doSample("counterfactual"); }

// System Help
function sampleSelf() { doSample("self"); }
function sampleFollowup() { doSample("followup"); }

// Legacy functions (kept for backwards compatibility)
function sampleDataDescription() { doSample("data"); }