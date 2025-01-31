document.getElementById("send-button").addEventListener("click", async function() {
    const userInput = document.getElementById("user-input").value;

    if (userInput.trim() !== "") {
        const chatWindow = document.getElementById("chat-window");
        chatWindow.innerHTML += `<div><strong>User:</strong> ${userInput}</div>`;

        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ query: userInput })
        });

        const data = await response.json();

        if (data.response) {
            chatWindow.innerHTML += `<div><strong>Chat:</strong> ${data.response}</div><br>`;
        } else {
            chatWindow.innerHTML += `<div><strong>Chat:</strong> Sorry, I couldn't understand your request.</div><br>`;
        }

        document.getElementById("user-input").value = "";
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
});
