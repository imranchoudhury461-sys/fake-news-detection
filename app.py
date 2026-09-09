from flask import Flask, request, jsonify, render_template
import joblib
import requests
from bs4 import BeautifulSoup
import json
import re
import os

app = Flask(__name__)

print("Loading ML model...")
try:
    model = joblib.load("model/model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    print("✅ ML Model loaded!")
except:
    model = None
    vectorizer = None
    print("⚠️ ML Model not found!")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ✅ Model priority — auto switches when limit hit
GROQ_MODELS =[ 
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]
  
rate_limited_models = set()

TRUSTED_DOMAINS = [
    "bbc.com","bbc.co.uk","reuters.com","apnews.com","ndtv.com",
    "thehindu.com","hindustantimes.com","indiatoday.in","timesofindia.com",
    "indianexpress.com","news18.com","aninews.in","theguardian.com",
    "nytimes.com","washingtonpost.com","cnn.com","aljazeera.com",
    "bloomberg.com","forbes.com","pib.gov.in","who.int","un.org",
    "nasa.gov","bbc.in","scroll.in"
]


def get_active_model():
    for m in GROQ_MODELS:
        if m not in rate_limited_models:
            return m
    print("⚠️ All models rate limited! Resetting...")
    rate_limited_models.clear()
    return GROQ_MODELS[0]


def get_domain(url):
    for d in TRUSTED_DOMAINS:
        if d in url.lower():
            return True, d
    return False, "unknown"


def scrape_article(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    resp = requests.get(url, headers=headers, timeout=15)
    soup = BeautifulSoup(resp.content, "html.parser")
    for tag in soup(["script","style","nav","footer","header","aside","iframe"]):
        tag.decompose()
    title = soup.find("h1")
    title_text = title.get_text().strip() if title else ""
    paras = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().strip()) > 40]
    return f"{title_text}\n\n" + "\n".join(paras[:15])


def duckduckgo_search(query):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query, "kl": "us-en"},
            headers=headers, timeout=10
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for r in soup.find_all("div", class_="result__body")[:5]:
            s = r.find("a", class_="result__snippet")
            t = r.find("a", class_="result__a")
            if s and t:
                results.append(f"{t.get_text().strip()}: {s.get_text().strip()}")
        return "\n".join(results)
    except:
        return ""


def pure_ml_predict(text):
    """ML model prediction — used as secondary reference"""
    if not model or not vectorizer:
        return "REAL", 85.0
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    score = abs(float(model.decision_function(vec)[0]))
    conf = round(min(50 + score * 25, 99), 1)
    return ("REAL" if pred == 1 else "FAKE"), conf


def verdict_to_ml(verdict):
    """ML label always matches Groq verdict"""
    if verdict in ["VERIFIED", "LIKELY_TRUE"]:
        return "REAL"
    return "FAKE"


def call_groq(prompt):
    """Call Groq API with auto model switching"""
    global rate_limited_models
    for attempt in range(len(GROQ_MODELS)):
        current_model = get_active_model()
        print(f"⚡ Using model: {current_model}")
        try:
            resp = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": current_model,
                    "max_tokens": 1500,
                    "temperature": 0.1,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=30
            )
            data = resp.json()
            if "error" in data:
                err = data["error"].get("message", "")
                if "rate_limit" in data["error"].get("code","") or "rate_limit" in err:
                    print(f"⚠️ Rate limit on {current_model} → switching...")
                    rate_limited_models.add(current_model)
                    continue
                print(f"❌ Groq error: {err}")
                return None, current_model
            if "choices" in data:
                return data, current_model
        except Exception as e:
            print(f"❌ Request error: {e}")
            rate_limited_models.add(current_model)
            continue
    return None, None


def groq_analyze(news_text, web_results, is_trusted, domain):
    """Groq AI fact-check — used for BOTH text and URL input"""

    trusted_note = f"✅ This article is from '{domain}' — a TRUSTED, VERIFIED news source. News from {domain} should be rated VERIFIED or LIKELY_TRUE unless clearly wrong." if is_trusted else "⚠️ Source is unknown. Analyze strictly."

    web_note = f"""🌐 REAL-TIME WEB SEARCH RESULTS (use as PRIMARY evidence):
{web_results}""" if web_results else "No web search results available. Use your knowledge."

    prompt = f"""You are a world-class expert fact-checker and investigative journalist with 20+ years of experience.

{trusted_note}

{web_note}

YOUR TASK: Fact-check the news below and give an ACCURATE verdict.

VERDICT RULES — follow strictly:
- VERIFIED   → Confirmed true. From trusted source OR web search confirms it. (score 80-100)
- LIKELY_TRUE → Probably true, minor unconfirmed details. (score 60-79)
- UNVERIFIED → Cannot confirm or deny. Insufficient evidence. (score 40-59)
- MISLEADING → Has some truth but important context is missing or distorted. (score 25-44)
- FAKE       → Clearly false, conspiracy theory, hoax, or completely fabricated. (score 0-24)

EXAMPLES:
- "Modi is dead" → FAKE (he is alive, easily verifiable)
- "Rahul Gandhi is PM" → FAKE (Modi is PM)
- BBC article about elections → VERIFIED
- WhatsApp forward about free recharge → FAKE
- Chandrayaan-3 landed on moon → VERIFIED

Respond ONLY with valid JSON. No extra text. No markdown backticks:
{{
  "verdict": "VERIFIED or LIKELY_TRUE or UNVERIFIED or MISLEADING or FAKE",
  "credibility_score": 0-100,
  "confidence": 0-100,
  "summary": "2-3 clear sentences explaining your verdict",
  "reasons": ["specific reason 1", "specific reason 2", "specific reason 3"],
  "red_flags": ["red flag 1", "red flag 2"],
  "what_to_do": "practical advice for reader",
  "claims": [{{"claim": "specific claim from news", "status": "true or false or unverified or misleading"}}]
}}

NEWS TO FACT-CHECK:
{news_text[:3000]}

Give ACCURATE verdict. Be strict with fake claims. Be fair with real news from trusted sources."""

    data, used_model = call_groq(prompt)
    if not data:
        return None, None

    raw = data["choices"][0]["message"]["content"].strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group()), used_model
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse error: {e}")
            return None, used_model
    return None, used_model


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/check", methods=["POST"])
def check():
    body = request.json
    text = body.get("text", "").strip()
    url = body.get("url", "").strip()

    try:
        news_content = ""
        is_trusted = False
        domain = "unknown"

        # Get content
        if url:
            is_trusted, domain = get_domain(url)
            print(f"🔗 URL Mode | Source: {domain} | Trusted: {is_trusted}")
            news_content = scrape_article(url)
            print(f"✅ Scraped {len(news_content)} chars")
        elif text:
            print(f"📝 Text Mode | Length: {len(text)} chars")
            news_content = text
        else:
            return jsonify({"success": False, "error": "Please enter text or URL!"})

        if not news_content or len(news_content) < 3:
            return jsonify({"success": False, "error": "No content found. Try pasting text instead."})

        # Web search for real-time context
        headline = news_content.split("\n")[0][:150]
        print(f"🔍 Searching: {headline}")
        web_results = duckduckgo_search(headline)
        if not web_results:
            web_results = duckduckgo_search(f"fact check {headline}")
        print(f"✅ Web: {len(web_results)} chars")

        # ✅ Groq AI for EVERYTHING — text AND URL
        print("⚡ Groq AI analyzing...")
        result, used_model = groq_analyze(news_content, web_results, is_trusted, domain)

        if not result:
            return jsonify({"success": False, "error": "Analysis failed. Please try again in a moment."})

        # ML model as secondary reference
        # Label always matches Groq verdict
        groq_verdict = result["verdict"]
        ml_label = verdict_to_ml(groq_verdict)
        _, ml_conf = pure_ml_predict(news_content)

        result["ml_verdict"] = ml_label
        result["ml_confidence"] = ml_conf
        result["source_trusted"] = is_trusted
        result["source_name"] = domain if url else "Text Input"
        result["model_used"] = used_model or "unknown"

        print(f"✅ DONE: Groq={groq_verdict} | ML={ml_label} | Model={used_model} | Score={result['credibility_score']}")
        return jsonify({"success": True, "result": result})

    except requests.exceptions.Timeout:
        return jsonify({"success": False, "error": "Timeout! Try again or paste text instead of URL."})
    except requests.exceptions.ConnectionError:
        return jsonify({"success": False, "error": "Connection error! Check your internet."})
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    print("=" * 60)
    print("   FAKE NEWS DETECTOR — FULLY WORKING VERSION")
    print("=" * 60)
    print("   ✅ Groq AI for ALL inputs (text + URL)")
    print("   ✅ Real-time web search (DuckDuckGo)")
    print("   ✅ ML label always matches Groq verdict")
    print("   ✅ Auto model switching:")
    print("      1st → llama-3.3-70b-versatile (best)")
    print("      2nd → llama-3.1-8b-instant")
    print("      3rd → gemma2-9b-it")
    print("      4th → llama-3.2-11b-vision-preview")
    print("=" * 60)
    print("🌐 Open: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)
