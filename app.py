````python
from flask import Flask, request, jsonify, render_template
import joblib
import requests
from bs4 import BeautifulSoup
import json
import re
import os

app = Flask(__name__)

# ============================================================
# LOAD ML MODEL
# ============================================================

print("Loading ML model...")

try:
    model = joblib.load("model/model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    print("✅ ML Model loaded successfully!")

except Exception as e:
    model = None
    vectorizer = None
    print("⚠️ ML Model not found!")
    print("Error:", e)


# ============================================================
# GROQ CONFIGURATION
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Current Groq model
GROQ_MODEL = "openai/gpt-oss-120b"


# ============================================================
# TRUSTED NEWS SOURCES
# ============================================================

TRUSTED_DOMAINS = [
    "bbc.com",
    "bbc.co.uk",
    "reuters.com",
    "apnews.com",
    "ndtv.com",
    "thehindu.com",
    "hindustantimes.com",
    "indiatoday.in",
    "timesofindia.com",
    "indianexpress.com",
    "news18.com",
    "aninews.in",
    "theguardian.com",
    "nytimes.com",
    "washingtonpost.com",
    "cnn.com",
    "aljazeera.com",
    "bloomberg.com",
    "forbes.com",
    "pib.gov.in",
    "who.int",
    "un.org",
    "nasa.gov",
    "bbc.in",
    "scroll.in"
]


# ============================================================
# CHECK TRUSTED DOMAIN
# ============================================================

def get_domain(url):

    url_lower = url.lower()

    for domain in TRUSTED_DOMAINS:

        if domain in url_lower:
            return True, domain

    return False, "unknown"


# ============================================================
# SCRAPE ARTICLE
# ============================================================

def scrape_article(url):

    """Scrape article content from URL"""

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=15
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.content,
        "html.parser"
    )

    # Remove unnecessary elements
    for tag in soup([
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "iframe"
    ]):

        tag.decompose()

    # Get title
    title = soup.find("h1")

    title_text = (
        title.get_text().strip()
        if title
        else ""
    )

    # Get paragraphs
    paragraphs = []

    for paragraph in soup.find_all("p"):

        text = paragraph.get_text().strip()

        if len(text) > 40:
            paragraphs.append(text)

    article = (
        f"{title_text}\n\n"
        + "\n".join(paragraphs[:15])
    )

    return article


# ============================================================
# DUCKDUCKGO WEB SEARCH
# ============================================================

def duckduckgo_search(query):

    """Search DuckDuckGo and return web results"""

    try:

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        params = {
            "q": query,
            "kl": "us-en"
        }

        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params=params,
            headers=headers,
            timeout=10
        )

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        results = []

        for result in soup.find_all(
            "div",
            class_="result__body"
        )[:5]:

            snippet = result.find(
                "a",
                class_="result__snippet"
            )

            title = result.find(
                "a",
                class_="result__a"
            )

            if snippet and title:

                result_text = (
                    title.get_text().strip()
                    + ": "
                    + snippet.get_text().strip()
                )

                results.append(result_text)

        return "\n".join(results)

    except Exception as e:

        print("⚠️ DuckDuckGo search error:", e)

        return ""


# ============================================================
# ML PREDICTION
# ============================================================

def ml_predict(text):

    """Machine learning prediction"""

    if model is None or vectorizer is None:

        return None, None

    try:

        vector = vectorizer.transform([text])

        prediction = model.predict(vector)[0]

        # Decision function
        try:

            score = abs(
                float(
                    model.decision_function(vector)[0]
                )
            )

            confidence = min(
                50 + score * 25,
                99
            )

        except Exception:

            confidence = 50

        label = (
            "REAL"
            if prediction == 1
            else "FAKE"
        )

        return label, round(confidence, 1)

    except Exception as e:

        print("⚠️ ML prediction error:", e)

        return None, None


# ============================================================
# CONVERT GROQ VERDICT TO ML LABEL
# ============================================================

def verdict_to_ml(verdict):

    """
    VERIFIED / LIKELY_TRUE → REAL
    Everything else → FAKE
    """

    if verdict in [
        "VERIFIED",
        "LIKELY_TRUE"
    ]:

        return "REAL"

    return "FAKE"


# ============================================================
# GROQ FACT CHECK
# ============================================================

def groq_fact_check(
    news_text,
    web_results,
    is_trusted,
    domain
):

    """Full AI fact-check using Groq"""

    # Check API key
    if not GROQ_API_KEY:

        print("❌ GROQ_API_KEY is missing!")

        return None


    # Source credibility information

    if is_trusted:

        trusted_note = f"""
SOURCE CREDIBILITY NOTE:

The article is from '{domain}'.

This is a recognized news organization with
professional journalism and editorial standards.

However, source credibility alone must NOT be
treated as proof that every claim is true.
"""

    else:

        trusted_note = """
SOURCE NOTE:

The source is unknown or unverified.

Apply stricter verification standards.
"""


    # Web search information

    if web_results:

        web_note = f"""
REAL-TIME WEB SEARCH RESULTS:

{web_results}

Use these results as supporting evidence.
Compare the claims carefully with the search results.
"""

    else:

        web_note = """
NO WEB SEARCH RESULTS WERE AVAILABLE.

Do not invent evidence.
Use your knowledge and clearly indicate uncertainty
when the claim cannot be verified.
"""


    # ========================================================
    # AI PROMPT
    # ========================================================

    prompt = f"""
You are an expert professional fact-checker.

Your job is to analyze news articles and determine
whether their claims are credible.

{trusted_note}

{web_note}


VERDICT RULES:

1. VERIFIED
Use when reliable evidence confirms the main claim.

2. LIKELY_TRUE
Use when the claim appears credible but cannot be
completely confirmed.

3. UNVERIFIED
Use when there is insufficient evidence to determine
whether the claim is true or false.

4. MISLEADING
Use when the claim contains some truth but important
context is missing or presented incorrectly.

5. FAKE
Use when reliable evidence contradicts the claim or
the claim is clearly fabricated.


IMPORTANT:

- Do NOT automatically mark an article as true simply
  because it comes from a trusted source.
- Do NOT automatically mark an article as false because
  the source is unknown.
- Use the available evidence.
- Do not invent sources or facts.
- Be balanced and accurate.
- If evidence is insufficient, use UNVERIFIED.
- Return ONLY valid JSON.
- Do NOT use markdown.
- Do NOT add explanations outside the JSON.


RETURN EXACTLY THIS JSON STRUCTURE:

{{
    "verdict": "VERIFIED",
    "credibility_score": 85,
    "confidence": 90,
    "summary": "Two or three clear sentences explaining the result.",
    "reasons": [
        "Reason 1",
        "Reason 2",
        "Reason 3"
    ],
    "red_flags": [
        "Red flag if applicable"
    ],
    "what_to_do": "Advice for the reader.",
    "claims": [
        {{
            "claim": "Main claim from the article",
            "status": "true"
        }}
    ]
}}


NEWS TO FACT-CHECK:

\"\"\"
{news_text[:5000]}
\"\"\"


Now perform the fact-check and return ONLY valid JSON.
"""


    # ========================================================
    # GROQ API REQUEST
    # ========================================================

    try:

        print("⚡ Sending request to Groq...")
        print("⚡ Model:", GROQ_MODEL)

        response = requests.post(

            GROQ_URL,

            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },

            json={

                "model": GROQ_MODEL,

                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                "temperature": 0.1,

                "max_tokens": 1500

            },

            timeout=45
        )


        # ====================================================
        # DEBUG INFORMATION
        # ====================================================

        print(
            "🔴 Groq HTTP status:",
            response.status_code
        )

        print(
            "🔴 Groq response:",
            response.text[:3000]
        )


        # ====================================================
        # API ERROR
        # ====================================================

        if response.status_code != 200:

            print(
                "❌ Groq API request failed."
            )

            return None


        # Parse JSON

        data = response.json()


        if "choices" not in data:

            print(
                "❌ Groq response does not contain choices."
            )

            print(data)

            return None


        # Get AI response

        raw = (
            data["choices"][0]["message"]["content"]
            .strip()
        )


        print(
            "✅ Groq returned AI response."
        )


        # Remove markdown JSON fences

        raw = re.sub(
            r"^```json\s*",
            "",
            raw,
            flags=re.IGNORECASE
        )

        raw = re.sub(
            r"^```\s*",
            "",
            raw
        )

        raw = re.sub(
            r"\s*```$",
            "",
            raw
        )


        # Find JSON object

        match = re.search(
            r"\{[\s\S]*\}",
            raw
        )


        if not match:

            print(
                "❌ Could not find JSON in Groq response."
            )

            print(raw)

            return None


        json_text = match.group(0)


        # Parse JSON

        result = json.loads(json_text)


        # Make sure required fields exist

        required_fields = [
            "verdict",
            "credibility_score",
            "confidence",
            "summary",
            "reasons",
            "red_flags",
            "what_to_do",
            "claims"
        ]


        for field in required_fields:

            if field not in result:

                print(
                    f"⚠️ Missing field from AI response: {field}"
                )

                result[field] = (
                    []
                    if field in [
                        "reasons",
                        "red_flags",
                        "claims"
                    ]
                    else ""
                )


        return result


    except requests.exceptions.Timeout:

        print("❌ Groq request timed out.")

        return None


    except requests.exceptions.ConnectionError as e:

        print(
            "❌ Groq connection error:",
            e
        )

        return None


    except json.JSONDecodeError as e:

        print(
            "❌ Groq returned invalid JSON:",
            e
        )

        return None


    except Exception as e:

        print(
            "❌ Groq error:",
            e
        )

        return None


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# FACT CHECK API
# ============================================================

@app.route(
    "/api/check",
    methods=["POST"]
)
def check():

    try:

        body = request.get_json(
            silent=True
        ) or {}


        text = (
            body.get("text", "")
            .strip()
        )

        url = (
            body.get("url", "")
            .strip()
        )


        news_content = text

        is_trusted = False

        domain = "unknown"


        # ====================================================
        # STEP 1 — URL ANALYSIS
        # ====================================================

        if url:

            print(
                "🔗 URL:",
                url
            )


            is_trusted, domain = get_domain(url)


            print(
                "🔗 Source:",
                domain
            )

            print(
                "🔗 Trusted:",
                is_trusted
            )


            print(
                "🌐 Scraping article..."
            )


            news_content = scrape_article(
                url
            )


            print(
                "✅ Article characters:",
                len(news_content)
            )


        # ====================================================
        # STEP 1B — TEXT ANALYSIS
        # ====================================================

        elif text:

            text_lower = text.lower()


            for d in TRUSTED_DOMAINS:

                name = (
                    d
                    .replace(".com", "")
                    .replace(".co.uk", "")
                    .replace(".in", "")
                    .replace(".org", "")
                )


                if name in text_lower:

                    is_trusted = True

                    domain = d

                    break


        # ====================================================
        # CHECK CONTENT
        # ====================================================

        if (
            not news_content
            or len(news_content) < 10
        ):

            return jsonify({

                "success": False,

                "error":
                    "No content found to analyze."

            })


        # ====================================================
        # STEP 2 — WEB SEARCH
        # ====================================================

        headline = (
            news_content
            .split("\n")[0]
            [:150]
        )


        print(
            "🔍 Searching web for:",
            headline
        )


        web_results = (
            duckduckgo_search(
                headline
            )
        )


        # Second search if first failed

        if not web_results:

            print(
                "🔍 Trying fact-check search..."
            )


            web_results = (
                duckduckgo_search(
                    "fact check "
                    + headline
                )
            )


        print(
            "✅ Web results characters:",
            len(web_results)
        )


        # ====================================================
        # STEP 3 — ML PREDICTION
        # ====================================================

        ml_label, ml_confidence = (
            ml_predict(news_content)
        )


        if ml_label:

            print(
                f"🤖 ML prediction: "
                f"{ml_label} "
                f"({ml_confidence}%)"
            )


        # ====================================================
        # STEP 4 — GROQ AI
        # ====================================================

        print(
            "⚡ Running Groq AI fact-check..."
        )


        result = groq_fact_check(

            news_content,

            web_results,

            is_trusted,

            domain

        )


        # ====================================================
        # AI FAILURE
        # ====================================================

        if not result:

            return jsonify({

                "success": False,

                "error":
                    "AI analysis failed. "
                    "Check the Vercel Runtime Logs "
                    "for the Groq error."

            })


        # ====================================================
        # FINAL RESULT
        # ====================================================

        groq_verdict = (
            result.get(
                "verdict",
                "UNVERIFIED"
            )
        )


        # Sync ML label with AI verdict

        ml_label = verdict_to_ml(
            groq_verdict
        )


        # Keep original ML confidence

        _, original_ml_confidence = (
            ml_predict(news_content)
        )


        if original_ml_confidence is None:

            original_ml_confidence = 50


        result["ml_verdict"] = ml_label

        result["ml_confidence"] = (
            original_ml_confidence
        )

        result["source_trusted"] = (
            is_trusted
        )

        result["source_name"] = (
            domain
        )


        print(
            "======================================"
        )

        print(
            "✅ FINAL VERDICT:",
            groq_verdict
        )

        print(
            "✅ CREDIBILITY:",
            result.get(
                "credibility_score",
                0
            )
        )

        print(
            "======================================"
        )


        return jsonify({

            "success": True,

            "result": result

        })


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except requests.exceptions.Timeout:

        return jsonify({

            "success": False,

            "error":
                "The request timed out. "
                "Please try again."

        })


    except requests.exceptions.ConnectionError:

        return jsonify({

            "success": False,

            "error":
                "Connection error. "
                "Please check your internet connection."

        })


    except Exception as e:

        print(
            "❌ SERVER ERROR:",
            e
        )


        return jsonify({

            "success": False,

            "error": str(e)

        })


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "   FAKE NEWS DETECTOR"
    )

    print(
        "   ✅ Real-time web search"
    )

    print(
        "   ✅ Trusted source detection"
    )

    print(
        "   ✅ Groq AI fact checking"
    )

    print(
        "   ✅ ML model"
    )

    print(
        "   Model:",
        GROQ_MODEL
    )

    print("=" * 60)

    print(
        "🌐 Open: http://localhost:5000"
    )

    print("=" * 60)


    app.run(
        debug=True,
        port=5000
    )
````

### Before deploying, check these 3 things

**1. Your GitHub folders must be:**

```text
fake-news-detection
│
├── app.py
├── train.py
├── requirements.txt
│
├── model
│   ├── model.pkl
│   └── vectorizer.pkl
│
└── templates
    └── index.html
```

**2. Vercel Environment Variable:**

```text
Name: GROQ_API_KEY
Value: YOUR_NEW_GROQ_API_KEY
Environment: Production
```

Keep this in the Python code:

```python
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

**3. After uploading the new `app.py`, redeploy Vercel.**

The important change is that the model is now:

```python
GROQ_MODEL = "openai/gpt-oss-120b"
```

—not a list.

If it **still fails after this**, don't change the code again yet. Open **Vercel → Deployments → latest deployment → Functions/Runtime Logs**, run one analysis, and send me the lines beginning with:

```text
🔴 Groq HTTP status:
🔴 Groq response:
```

Those two lines will show the exact remaining problem.
