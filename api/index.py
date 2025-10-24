from flask import Flask, request, jsonify, Response
import requests
import json
import uuid

app = Flask(__name__)

T3_COOKIE = "COOKIE HERE !!!!" # CHANGE THIS !!!!!
MODEL = "gpt-5-chat"
# gpt-5-chat,
# claude-4.5-haiku-reasoning,
# etc (you can see the api model term in dev tools when u send a message in t3.chat)

def t3_req(messages, model, stream=False):
    headers = {"cookie": T3_COOKIE} # only header they use for auth, you can find your cookie by looking in the network tab in dev tools when u send a message in t3.chat
    
    t3_messages = [
        {
            "parts": [{"text": msg["content"], "type": "text"}],
            "role": msg["role"],
            "attachments": []
        }
        for msg in messages
    ]
    
    payload = {
        "messages": t3_messages,
        "threadMetadata": {"id": str(uuid.uuid4()), "title": "Title"},
        "responseMessageId": str(uuid.uuid4()),
        "model": model,
        "convexSessionId": str(uuid.uuid4()),
        "modelParams": {"reasoningEffort": "low", "includeSearch": False}, # change reasoning accordingly. (you can see in dev tools which reasoning effort is used)
        "preferences": {"name": "", "occupation": "", "selectedTraits": [], "additionalInfo": ""},
        "userInfo": {"timezone": "America/Los_Angeles"}
    }
    # ^^^ change thread metadata, response message id, and convex session id to the correct values if you want it to appear in ur t3.chat history.
    
    r = requests.post("https://t3.chat/api/chat", headers=headers, json=payload, stream=True)
    
    full_text = ""
    for line in r.iter_lines():
        if not line or not line.startswith(b"data: "): # t3.chat's response format cuz they do stream
            continue
        try:
            data = json.loads(line[6:].decode("utf-8"))
            if data.get("type") == "text-delta":
                delta = data.get("delta", "")
                full_text += delta
                if stream:
                    yield delta
        except Exception:
            continue
    
    if not stream:
        yield full_text

@app.route('/v1/chat/completions', methods=['POST'])
def chat_comp():
    data = request.json
    messages = data.get('messages', [])
    model = data.get('model', MODEL)
    stream = data.get('stream', False)
    
    if stream:
        def generate():
            for chunk in t3_req(messages, model, stream=True):
                response_chunk = {
                    "choices": [{"delta": {"content": chunk}, "index": 0}],
                    "model": model,
                    "object": "text_completion.chunk"
                }
                yield f"data: {json.dumps(response_chunk)}\n\n"
        
        return Response(generate(), mimetype='text/event-stream')
    else:
        full_text = next(t3_req(messages, model, stream=False)) # next cuz of yields
        return jsonify({
            "choices": [{"message": {"content": full_text, "role": "assistant"}, "index": 0, "finish_reason": "stop"}],
            "model": model,
            "object": "text_completion"
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)