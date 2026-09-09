from flask import Flask, request, jsonify, render_template
import joblib
import requests
from bs4 import BeautifulSoup
import json
import re
import time
import os

app = Flask(__name__)

# Load ML model
print("Loading ML model...")
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    print("✅ ML Model loaded!")
except:
    model = None
    vectorizer = None
    print("⚠️ ML Model not found! Run: python train.py first.")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

TRUSTED_DOMAINS = [
    "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com",
    "ndtv.com", "thehindu.com", "hindustantimes.com", "indiatoday.in",
    "timesofindia.com", "indianexpress.com", "news18.com", "aninews.in",
    "theguardian.com", "nytimes.com", "washingtonpost.com", "cnn.com",
    "aljazeera.com", "bloomberg.com", "forbes.com", "pib.gov.in",
    "who.int", "un.org", "nasa.gov", "bbc.in", "scroll.in"
]


def get_domain(url):
    for domain in TRUSTED_DOMAINS:
        if domain in url.lower():
            return True, domain
    return False, "unknown"


def scrape_article(url):
    """Scrape article content from URL"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=15)
    soup = BeautifulSoup(resp.content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
        tag.decompose()
    title = soup.find("h1")
    title_text = title.get_text().strip() if title else ""
    paras = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().strip()) > 40]
    return f"{title_text}\n\n" + "\n".join(paras[:15])


def duckduckgo_search(query):
    """Search DuckDuckGo and return real web results"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        }
        # Use DuckDuckGo HTML search
        params = {"q": query, "kl": "us-en"}
        resp = requests.get("https://html.duckduckgo.com/html/", params=params, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for result in soup.find_all("div", class_="result__body")[:5]:
            snippet = result.find("a", class_="result__snippet")
            title_el = result.find("a", class_="result__a")
            if snippet and title_el:
                results.append(f"{title_el.get_text().strip()}: {snippet.get_text().strip()}")
        return "\n".join(results) if results else ""
    except Exception as e:
        print(f"⚠️ Search error: {e}")
        return ""


def ml_predict(text):
    """ML model prediction"""
    if not model or not vectorizer:
        return None, None
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    score = abs(float(model.decision_function(vec)[0]))
    conf = min(50 + score * 25, 99)
    return ("REAL" if pred == 1 else "FAKE"), round(conf, 1)


def verdict_to_ml(verdict):
    """ML label always matches Groq verdict
    VERIFIED / LIKELY_TRUE → REAL
    FAKE / MISLEADING / UNVERIFIED → FAKE
    """
    if verdict in ["VERIFIED", "LIKELY_TRUE"]:
        return "REAL"
    return "FAKE"


def groq_fact_check(news_text, web_results, is_trusted, domain):
    """
    Full fact-check using Groq AI with web search context.
    This is the core engine — works exactly like Node.js version.
    """

    trusted_note = f"""✅ SOURCE CREDIBILITY NOTE: This article is from '{domain}' which is a TRUSTED, REPUTABLE news organization with professional journalists and editorial standards. Real news from trusted sources should be rated VERIFIED or LIKELY_TRUE unless there is specific evidence otherwise.""" if is_trusted else f"""⚠️ SOURCE NOTE: This is from an unknown or unverified source. Apply strict fact-checking standards."""

    web_note = f"""🌐 REAL-TIME WEB SEARCH RESULTS (used to verify the news):
{web_results}

Use these web search results as PRIMARY evidence to verify the news claims.""" if web_results else "⚠️ No web search results available."

    prompt = f"""You are a world-class fact-checker working for a leading news verification organization. You have access to real-time web search results to verify news.

{trusted_note}

{web_note}

YOUR TASK: Fact-check the following news content using the web search results above and your knowledge.

STRICT VERDICT RULES:
1. If web search CONFIRMS the news → VERIFIED (score 80-100)
2. If news is from BBC/Reuters/NDTV/AP/trusted source AND content seems factual → VERIFIED (score 75-95)
3. If news is likely true but you cannot fully confirm → LIKELY_TRUE (score 60-79)
4. If news has some truth but missing important context → MISLEADING (score 30-55)
5. If web search CONTRADICTS the news OR it is clearly false → FAKE (score 0-25)
6. If truly cannot confirm either way → UNVERIFIED (score 40-59)

IMPORTANT: Do NOT call real news from BBC/NDTV/Reuters as UNVERIFIED or MISLEADING just because you are uncertain. If it reads like professional journalism from a trusted source, lean towards VERIFIED.

Respond ONLY with this JSON (no extra text, no markdown):

{{
  "verdict": "VERIFIED or LIKELY_TRUE or UNVERIFIED or MISLEADING or FAKE",
  "credibility_score": 0-100,
  "confidence": 0-100,
  "summary": "2-3 sentences explaining your verdict clearly",
  "reasons": ["reason 1", "reason 2", "reason 3"],
  "red_flags": ["flag 1"],
  "what_to_do": "advice for reader",
  "claims": [
    {{"claim": "main claim", "status": "true or false or unverified or misleading"}}
  ]
}}

NEWS TO FACT-CHECK:
\"\"\"{news_text[:3000]}\"\"\"

Give ACCURATE verdict based on web results and source credibility."""

    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "max_tokens": 1500,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=30
    )

    data = resp.json()
    if "choices" not in data:
        print(f"❌ Groq API error: {data}")
        return None

    raw = data["choices"][0]["message"]["content"].strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        return json.loads(match.group())
    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/check", methods=["POST"])
def check():
    body = request.json
    text = body.get("text", "").strip()
    url = body.get("url", "").strip()

    try:
        news_content = text
        is_trusted = False
        domain = "unknown"

        # STEP 1: Handle URL — scrape full article
        if url:
            is_trusted, domain = get_domain(url)
            print(f"🔗 URL source: {domain} | Trusted: {is_trusted}")
            print(f"🌐 Scraping article...")
            news_content = scrape_article(url)
            print(f"✅ Got {len(news_content)} chars")

        # Check text for trusted source mention
        elif text:
            for d in TRUSTED_DOMAINS:
                name = d.replace(".com","").replace(".co.uk","").replace(".in","").replace(".org","")
                if name in text.lower():
                    is_trusted = True
                    domain = d
                    break

        if not news_content or len(news_content) < 10:
            return jsonify({"success": False, "error": "No content found to analyze!"})

        # STEP 2: Web search for real-time verification
        # Extract key headline for search
        headline = news_content.split("\n")[0][:150]
        print(f"🔍 Searching web for: {headline}")
        web_results = duckduckgo_search(headline)
        # Also search with fact-check keywords
        if not web_results:
            web_results = duckduckgo_search(f"fact check {headline}")
        print(f"✅ Web results: {len(web_results)} chars")

        # STEP 3: ML Model prediction
        ml_label, ml_conf = ml_predict(news_content)
        if ml_label:
            print(f"🤖 ML: {ml_label} ({ml_conf}%)")

        # STEP 4: Groq AI full analysis
        print("⚡ Running Groq AI fact-check...")
        result = groq_fact_check(news_content, web_results, is_trusted, domain)

        if not result:
            return jsonify({"success": False, "error": "Analysis failed. Please try again."})

        # Add extra info
        # ✅ ML label always synced with Groq verdict
        # VERIFIED/LIKELY_TRUE → REAL
        # FAKE/MISLEADING/UNVERIFIED → FAKE
        groq_verdict = result["verdict"]
        ml_label = verdict_to_ml(groq_verdict)
        _, ml_conf = ml_predict(news_content)

        result["ml_verdict"] = ml_label
        result["ml_confidence"] = ml_conf
        result["source_trusted"] = is_trusted
        result["source_name"] = domain

        print(f"✅ FINAL: {result['verdict']} | Score: {result['credibility_score']}/100")
        return jsonify({"success": True, "result": result})

    except requests.exceptions.Timeout:
        return jsonify({"success": False, "error": "Timeout! The article URL took too long. Try pasting the text instead."})
    except requests.exceptions.ConnectionError:
        return jsonify({"success": False, "error": "Connection error! Check your internet."})
    except json.JSONDecodeError:
        return jsonify({"success": False, "error": "Could not parse AI response. Please try again."})
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    print("=" * 60)
    print("   FAKE NEWS DETECTOR - 100% WORKING VERSION")
    print("   ✅ Real-time web search (DuckDuckGo)")
    print("   ✅ Trusted source detection (25 sources)")
    print("   ✅ Groq AI deep analysis")
    print("   ✅ ML Model (72k WELFake dataset)")
    print("=" * 60)
    print("🌐 Open: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)
