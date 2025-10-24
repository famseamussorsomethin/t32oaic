from flask import Flask, request, jsonify, Response
import requests
import json
import uuid

app = Flask(__name__)

T3_COOKIE = "COOKIE HERE!!!" # this is serious mode. put the cookie here nOW!
# models:
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
    
    textresp = ""
    reasoningresp = ""
    for line in r.iter_lines():
        if not line or not line.startswith(b"data: "): # t3.chat's response format cuz they do stream for good user experience
            continue
        try:
            data = json.loads(line[6:].decode("utf-8"))
            event_type = data.get("type")
            
            if event_type == "reasoning-delta": # this is the actual reasoning stuff in t3 chat api, it starts with reasoning start and ends with reasoning end
                delta = data.get("delta", "")
                reasoningresp += delta
                if stream:
                    yield ("reasoning", delta)
            elif event_type == "text-delta":
                delta = data.get("delta", "")
                textresp += delta
                if stream:
                    yield ("text", delta)
        except Exception:
            continue
    
    if not stream:
        if reasoningresp:
            textresp = f"<think>\n{reasoningresp}\n</think>\n\n{textresp}"
        yield textresp

@app.route('/v1/chat/completions', methods=['POST'])
def chat_comp():
    data = request.json
    messages = data.get('messages', [])
    model = data.get('model')
    stream = data.get('stream', False)
    
    if stream:
        def generate():
            reasoning_stream = ""
            reasoning_sent = False
            
            for text_type, content in t3_req(messages, model, stream=True): # text type checks if its reasoning or text
                if text_type == "reasoning":
                    reasoning_stream += content
                elif text_type == "text":
                    if not reasoning_sent and reasoning_stream:
                        fullresp_chunk = {
                            "choices": [{"delta": {"content": f"<think>\n{reasoning_stream}\n</think>\n\n"}, "index": 0}],
                            "model": model,
                            "object": "text_completion.chunk"
                        }
                        yield f"data: {json.dumps(fullresp_chunk)}\n\n"
                        reasoning_sent = True
                    
                    fullresp_chunk = {
                        "choices": [{"delta": {"content": content}, "index": 0}],
                        "model": model,
                        "object": "text_completion.chunk"
                    }
                    yield f"data: {json.dumps(fullresp_chunk)}\n\n"
        
        return Response(generate(), mimetype='text/event-stream')
    else:
        textresp = next(t3_req(messages, model, stream=False)) # next cuz of yields
        return jsonify({
            "choices": [{"message": {"content": textresp, "role": "assistant"}, "index": 0, "finish_reason": "stop"}],
            "model": model,
            "object": "text_completion"
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)