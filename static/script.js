// Set per plan by the server. Only used to resume a plan
// that is paused at the approval gate — a new request
// always starts a new thread.
let currentThreadId = null;
let latestAnswerMarkdown = "";

function setPrompt(text) {
    document.getElementById("userInput").value = text;
}

function setLoading(isLoading) {
    const sendBtn = document.getElementById("sendBtn");
    const btnText = document.getElementById("btnText");
    const btnLoader = document.getElementById("btnLoader");

    sendBtn.disabled = isLoading;

    if (isLoading) {
        btnText.classList.add("hidden");
        btnLoader.classList.remove("hidden");
    } else {
        btnText.classList.remove("hidden");
        btnLoader.classList.add("hidden");
    }
}

function showError(message) {
    const errorBox = document.getElementById("errorBox");

    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
}

function hideError() {
    const errorBox = document.getElementById("errorBox");

    errorBox.classList.add("hidden");
    errorBox.textContent = "";
}

function renderMarkdown(element, text) {
    if (typeof marked !== "undefined") {
        element.innerHTML = marked.parse(text);
    } else {
        element.innerText = text;
    }
}

function scrollTo(section) {
    section.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}

function hideApproval() {
    document.getElementById("approvalSection").classList.add("hidden");
}

function showResult(answer, threadId) {
    latestAnswerMarkdown = answer;

    hideApproval();

    const resultSection = document.getElementById("resultSection");
    const resultBox = document.getElementById("resultBox");
    const threadInfo = document.getElementById("threadInfo");

    renderMarkdown(resultBox, answer);

    threadInfo.textContent = `Thread ID: ${threadId}`;

    resultSection.classList.remove("hidden");

    scrollTo(resultSection);
}

function showApproval(draft, revisionCount) {
    const approvalSection = document.getElementById("approvalSection");
    const draftBox = document.getElementById("draftBox");
    const hint = document.getElementById("approvalHint");
    const feedbackInput = document.getElementById("feedbackInput");

    renderMarkdown(draftBox, draft);

    feedbackInput.value = "";

    hint.textContent = revisionCount > 0
        ? `Revision ${revisionCount}. Approve this draft, or ask for more changes.`
        : "Approve this draft to generate the final plan, or tell us what to change.";

    // The final plan from a previous run is no longer current.
    document.getElementById("resultSection").classList.add("hidden");

    approvalSection.classList.remove("hidden");

    scrollTo(approvalSection);
}

// Both endpoints return the same shape.
function handlePlanResponse(data) {
    currentThreadId = data.thread_id;

    if (data.status === "awaiting_approval") {
        showApproval(data.draft_itinerary, data.revision_count || 0);
        return;
    }

    if (data.status === "rejected") {
        hideApproval();
        showError(data.rejection_reason || data.answer || "Request rejected.");
        return;
    }

    showResult(data.answer, data.thread_id);
}

async function sendMessage() {
    hideError();

    const input = document.getElementById("userInput");
    const message = input.value.trim();

    if (!message) {
        showError("Please enter your travel request first.");
        return;
    }

    setLoading(true);

    try {
        const response = await fetch("/api/travel", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message,
                thread_id: currentThreadId
            })
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || "Something went wrong.");
        }

        handlePlanResponse(data);

    } catch (error) {
        showError(error.message);
    } finally {
        setLoading(false);
    }
}

function setApprovalLoading(isLoading) {
    document.getElementById("approveBtn").disabled = isLoading;
    document.getElementById("reviseBtn").disabled = isLoading;
}

async function submitApproval(action) {
    hideError();

    const feedback = document.getElementById("feedbackInput").value.trim();

    if (action === "revise" && !feedback) {
        showError("Tell us what you'd like changed first.");
        return;
    }

    if (!currentThreadId) {
        showError("This plan has expired. Please generate a new one.");
        return;
    }

    setApprovalLoading(true);
    setLoading(true);

    try {
        const response = await fetch("/api/travel/approve", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                thread_id: currentThreadId,
                action: action,
                feedback: feedback
            })
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || "Something went wrong.");
        }

        handlePlanResponse(data);

    } catch (error) {
        showError(error.message);
    } finally {
        setApprovalLoading(false);
        setLoading(false);
    }
}

function copyResult() {
    const resultBox = document.getElementById("resultBox");
    const text = resultBox.innerText;

    if (!text) {
        return;
    }

    navigator.clipboard.writeText(text)
        .then(() => {
            const copyBtn = document.querySelector(".copy-btn");
            const oldText = copyBtn.textContent;

            copyBtn.textContent = "Copied!";

            setTimeout(() => {
                copyBtn.textContent = oldText;
            }, 1400);
        })
        .catch(() => {
            showError("Could not copy result.");
        });
}

function downloadPDF() {
    const pdfContent = document.getElementById("pdfContent");

    if (!latestAnswerMarkdown || !pdfContent) {
        showError("No travel plan available to download.");
        return;
    }

    const downloadBtn = document.querySelector(".download-btn");
    const oldText = downloadBtn.textContent;

    downloadBtn.textContent = "Preparing PDF...";
    downloadBtn.disabled = true;

    const options = {
        margin: 0.5,
        filename: "ai-travel-plan.pdf",
        image: {
            type: "jpeg",
            quality: 0.98
        },
        html2canvas: {
            scale: 2,
            useCORS: true,
            backgroundColor: "#ffffff"
        },
        jsPDF: {
            unit: "in",
            format: "a4",
            orientation: "portrait"
        },
        pagebreak: {
            mode: ["avoid-all", "css", "legacy"]
        }
    };

    html2pdf()
        .set(options)
        .from(pdfContent)
        .save()
        .then(() => {
            downloadBtn.textContent = oldText;
            downloadBtn.disabled = false;
        })
        .catch(() => {
            downloadBtn.textContent = oldText;
            downloadBtn.disabled = false;
            showError("Could not download PDF.");
        });
}

document.addEventListener("keydown", function(event) {
    if (event.ctrlKey && event.key === "Enter") {
        sendMessage();
    }
});