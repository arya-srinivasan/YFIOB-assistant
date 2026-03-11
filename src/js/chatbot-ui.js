const API_URL = "http://localhost:8000";

let userId         = localStorage.getItem("yfiob_user_id") || "anonymous";
let studentContext = {};

export async function getBotResponse(message) {
    const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            message,
            user_id:         userId,
            student_context: studentContext,
        })
    });

    const data = await response.json();

    // Keep student context updated between messages
    if (data.student_context) {
        studentContext = data.student_context;
    }

    return data.response;
}