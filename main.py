import os
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from authlib.integrations.starlette_client import OAuth, OAuthError
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from starlette.middleware.sessions import SessionMiddleware
from datetime import datetime
from bson import ObjectId # Import ObjectId for explicit conversion
import json
from groq import Groq  # IMPORT GROQ
from pydantic import BaseModel

# 1. Load Config
load_dotenv()

# 2. App Setup
app = FastAPI()

# Add Session Middleware (Required for OAuth)
# WARNING: Ensure SECRET_KEY is set in your .env file
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY"))

# 3. Path Setup (Connecting to your sibling Frontend folder)
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "Frontend"

if not FRONTEND_DIR.exists():
    raise RuntimeError(f"Frontend directory not found at {FRONTEND_DIR}")

# 4. Mount Static Files
# This makes style.css, script.js, and images available to the browser
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
app.mount("/image", StaticFiles(directory=str(FRONTEND_DIR / "image")), name="images")
# 5. Database Setup (MongoDB Atlas)
@app.on_event("startup")
async def startup_db_client():
    mongo_url = os.getenv("MONGODB_URL")
    if not mongo_url:
        print("❌ ERROR: MONGODB_URL is missing in .env")
        return
    app.mongodb_client = AsyncIOMotorClient(mongo_url)
    app.mongodb = app.mongodb_client[os.getenv("DB_NAME")]
    print("✅ Connected to MongoDB Atlas (Online)")

@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()

# 6. Google OAuth Setup
oauth = OAuth()
oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# --- ROUTES ---

# Route 1: The Entry Point (Login Form)
@app.get("/")
async def read_root():
    return FileResponse(FRONTEND_DIR / "loginform.html")

# Route 2: Start Google Login
@app.get('/login/google')
async def login_google(request: Request):
    redirect_uri = request.url_for('auth_google_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)

# Route 3: Google Callback (The Magic Happens Here)
@app.get('/auth/google/callback')
async def auth_google_callback(request: Request):
    try:
        # 1. Get Token & User Info from Google
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get('userinfo')
        
        if not user_info:
            raise HTTPException(status_code=400, detail="Failed to get user info")

        # 2. Save User to Online MongoDB Atlas
        users_collection = app.mongodb["users"]
        user_data = {
            "google_id": user_info.get("sub"),
            "email": user_info.get("email"),
            "name": user_info.get("name"),
            "picture": user_info.get("picture"),
            "last_login": datetime.utcnow().isoformat()
        }
        
        # Update if exists, Insert if new (Upsert)
        await users_collection.update_one(
            {"email": user_data["email"]},
            {"$set": user_data},
            upsert=True
        )

        # 3. Store user in session (cookie) so they stay logged in
        request.session['user'] = user_data

        # 4. Redirect to the Dashboard
        return RedirectResponse(url='/dashboard')

    except OAuthError as e:
        return {"error": f"OAuth Error: {e.error}"}

# Route 4: The Dashboard (Index.html)
@app.get("/dashboard")
async def dashboard(request: Request):
    user = request.session.get('user')
    if not user:
        return RedirectResponse(url='/') # Kick back to login if not signed in
        
    return FileResponse(FRONTEND_DIR / "index.html")

# Helper to serve other files if needed (like images referenced in HTML)
@app.get("/{filename}")
async def serve_root_files(filename: str):
    file_path = FRONTEND_DIR / filename
    if file_path.is_file():
        return FileResponse(file_path)
    return {"error": "File not found"}


# --- USER PROFILE ROUTE ---
@app.get("/api/me")
async def get_current_user(request: Request):
    user = request.session.get('user')
    if not user:
        return {"error": "Not logged in"}
    return user


@app.post("/api/campaigns/create")
async def create_new_campaign_api(request: Request):
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")

    try:
        data = await request.json()
        campaign_name = data.get("name", "Untitled Campaign")
        # 1. Capture the type (default to 'audio' if not sent)
        campaign_type = data.get("type", "audio") 
    except Exception:
        campaign_name = "Untitled Campaign"
        campaign_type = "audio"

    new_campaign = {
        "user_id": user["google_id"],
        "name": campaign_name,
        "type": campaign_type, # 2. Save type to DB
        "created_at": datetime.utcnow().isoformat(),
        "status": "In Design",
        "candidate_count": 0,
        "config": {
            "mode": "Technical Round",
            "duration": "12 Mins",
            "script": "Initial script prompt...",
            "voice": "Sarah",
        }
    }

    campaigns_collection = app.mongodb["campaigns"]
    result = await campaigns_collection.insert_one(new_campaign)
    
    new_campaign["_id"] = str(new_campaign["_id"]) 

    return {
        "message": "Campaign created successfully",
        "id": str(result.inserted_id),
        "campaign": new_campaign
    }
# --- CAMPAIGN RETRIEVAL ROUTE ---
@app.get("/api/campaigns")
async def get_user_campaigns(request: Request):
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")

    campaigns_collection = app.mongodb["campaigns"]
    
    # Fetch all campaigns belonging to the logged-in user
    campaigns_cursor = campaigns_collection.find({"user_id": user["google_id"]}).sort("created_at", -1)
    
    campaigns_list = []
    async for doc in campaigns_cursor:
        # Crucial for GET route: Convert MongoDB ObjectId to string for JSON serialization
        doc["id"] = str(doc.pop("_id"))
        campaigns_list.append(doc)
        
    return {"campaigns": campaigns_list}



# --- Add these imports at the top of main.py ---
from fastapi import UploadFile, File, Form
import pandas as pd
import io
from pypdf import PdfReader

# --- IN main.py ---
# Replace your existing '/api/parse-candidates' with this smarter version

@app.post("/api/parse-candidates")
async def parse_candidates(file: UploadFile = File(...)):
    contents = await file.read()
    
    try:
        # 1. Load Data into Pandas
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return {"error": "Unsupported file format"}
        
        # 2. Smart Column Matcher
        # We look for these keywords in the Excel headers
        def get_column_by_keyword(df_columns, keywords):
            # Normalization helper
            normalize = lambda x: str(x).lower().replace("_", "").replace(" ", "").strip()
            
            for col in df_columns:
                col_norm = normalize(col)
                for kw in keywords:
                    if kw in col_norm:
                        return col
            return None

        # Define variations of keywords
        name_col = get_column_by_keyword(df.columns, ["name", "candidate", "fullname", "student"])
        email_col = get_column_by_keyword(df.columns, ["email", "mail", "gmail", "e-mail"])
        phone_col = get_column_by_keyword(df.columns, ["phone", "mobile", "contact", "cell", "number", "tel"])

        if not name_col and not email_col and not phone_col:
            return {"error": "Could not automatically identify Name, Email, or Phone columns."}

        # 3. Extraction & Cleaning Loop
        candidates = []
        
        # Convert NaN to None for easier handling
        df = df.where(pd.notnull(df), None)

        for _, row in df.iterrows():
            # Get raw values
            name_val = row[name_col] if name_col else "Unknown Candidate"
            email_val = row[email_col] if email_col else ""
            phone_val = row[phone_col] if phone_col else None

            # --- CRITICAL: SKIP IF PHONE IS MISSING ---
            # The user requested: "if empty value like mobile then don't take this"
            if not phone_val or str(phone_val).strip() == "":
                continue 

            # Clean Phone Number (Remove decimals like 9999.0 generated by Excel)
            phone_str = str(phone_val).split('.')[0].strip()

            candidates.append({
                "name": str(name_val).strip(),
                "email": str(email_val).strip(),
                "phone": phone_str
            })
            
        return {"candidates": candidates, "count": len(candidates)}
        
    except Exception as e:
        print(f"Error parsing file: {e}")
        return {"error": "Failed to parse file. Please check format."}

@app.post("/api/generate-questions")
async def generate_questions_api(
    job_role: str = Form(""),       # NEW FIELD
    context_text: str = Form(""),
    file: UploadFile = File(None)
):
    # 1. Parse PDF if exists
    pdf_text = ""
    if file:
        try:
            reader = PdfReader(io.BytesIO(await file.read()))
            for page in reader.pages:
                pdf_text += page.extract_text()
        except:
            pass

    # 2. Smart Prompt Construction (Mock AI)
    # In a real app, you would send this 'full_prompt' to OpenAI/Gemini
    
    questions = []
    
    # Dynamic header based on role
    role_title = job_role if job_role else "Candidate"
    questions.append(f"--- Interview Script for: {role_title} ---")
    questions.append("Objective: Evaluate technical skills and cultural fit.\n")

    # Generate context-aware questions
    if "python" in context_text.lower() or "python" in job_role.lower():
        questions.append("1. [Technical] Explain the difference between list and tuple in Python.")
        questions.append("2. [Scenario] How do you handle memory management in large datasets?")
    elif "sales" in job_role.lower():
        questions.append("1. [Behavioral] Describe a time you turned a 'No' into a 'Yes'.")
        questions.append("2. [Strategy] How do you prioritize your lead pipeline?")
    else:
        # Generic fallback
        questions.append(f"1. Can you walk us through your experience as a {role_title}?")
        questions.append("2. What is the most challenging project you've worked on recently?")

    questions.append("3. [Culture] How do you handle disagreements with team members?")
    questions.append("4. Do you have any questions for us about the role?")

    return {"questions": "\n".join(questions)}


# --- UPDATE THIS FUNCTION IN main.py ---
@app.post("/api/candidates/save")
async def save_candidates(request: Request):
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")

    payload = await request.json()
    campaign_id = payload.get("campaign_id")
    candidates = payload.get("candidates", [])
    
    if not campaign_id:
        raise HTTPException(status_code=400, detail="campaign_id is required")

    coll = app.mongodb["candidates"]
    
    # 1. Get existing phone numbers for this campaign to avoid duplicates
    existing_cursor = coll.find(
        {"campaign_id": campaign_id, "user_id": user["google_id"]},
        {"phone": 1}
    )
    existing_phones = set()
    async for doc in existing_cursor:
        if "phone" in doc:
            existing_phones.add(doc["phone"])

    # 2. Filter out duplicates
    new_candidates = []
    skipped_count = 0

    for c in candidates:
        clean_phone = str(c.get("phone", "")).strip()
        
        # Validation: Must have phone and NOT be in existing list
        if clean_phone and clean_phone not in existing_phones:
            c["campaign_id"] = campaign_id
            c["user_id"] = user["google_id"]
            c["created_at"] = datetime.utcnow().isoformat()
            c["status"] = "Pending"
            c["phone"] = clean_phone # Ensure clean phone is saved
            new_candidates.append(c)
            existing_phones.add(clean_phone) # Add to set to prevent dups within the same batch
        else:
            skipped_count += 1
    
    insert_count = 0
    if new_candidates:
        try:
            result = await coll.insert_many(new_candidates)
            insert_count = len(result.inserted_ids)
        except Exception as e:
            print(f"Database Insert Error: {e}")
            raise HTTPException(status_code=500, detail="Failed to save candidates")
            
    return {
        "message": "Sync complete",
        "added": insert_count,
        "skipped": skipped_count
    }

# --- ADD THIS NEW ENDPOINT TO main.py ---
@app.delete("/api/candidates/{candidate_id}")
async def delete_single_candidate(candidate_id: str, request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)

    try:
        from bson import ObjectId
        res = await app.mongodb["candidates"].delete_one({
            "_id": ObjectId(candidate_id),
            "user_id": user["google_id"]
        })
        
        if res.deleted_count == 1:
            return {"status": "success"}
        raise HTTPException(status_code=404, detail="Candidate not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# --- GROQ (LLAMA 3) SETUP ---
# Initialize Groq Client
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)



class BlueprintRequest(BaseModel):
    company_name: str
    job_role: str
    description: str
    candidate_count: int
    agent_persona: str
    strictness: str
    interview_mode: str
    duration: int

@app.post("/api/generate-blueprint")
async def generate_blueprint_api(request: Request):
    try:
        data = await request.json()

        # 1. Extract Data
        company = data.get('company_name', 'TechCorp')
        role = data.get('job_role', 'Candidate')
        mode = data.get('interview_mode', 'Mixed')
        strictness = data.get('strictness', 'Balanced')
        
        # Deep Context
        domain = data.get('domain', 'General')
        tech_stack = data.get('tech_stack', 'General Skills')
        hr_focus = data.get('hr_focus', 'Communication')
        job_desc = data.get('job_description', '')
        
        # Agent Persona Settings
        agent_name = data.get('agent_name', 'Interviewer')
        language = data.get('language', 'English')

        # 2. Build the "Persona Block" (Who is the AI?)
        persona_prompt = f"""
        IDENTITY: You are {agent_name}, a professional AI Recruiter for {company}.
        ROLE: You are interviewing a candidate for the position of {role}.
        LANGUAGE: Conduct the interview in {language}.
        """

        # 3. Build the "Behavior Block" based on Strictness
        if "Strict" in strictness:
            behavior_prompt = "BEHAVIOR: You are skeptical and rigorous. Do not accept vague answers. If the candidate mentions a keyword, ask 'Why?' or 'How?'. Drill down into specific implementation details. If they struggle, move on without helping."
        elif "Friendly" in strictness:
            behavior_prompt = "BEHAVIOR: You are warm, encouraging, and supportive. If the candidate struggles, offer a small hint. Focus on their potential rather than just right/wrong answers."
        else: # Balanced
            behavior_prompt = "BEHAVIOR: Be professional and neutral. Ask follow-up questions to verify depth, but keep the conversation moving smoothly. Use the STAR method to guide them."

        # 4. Build the "Knowledge Base" (The System Problem/Context)
        knowledge_prompt = ""
        
        if "Technical" in mode or "Mixed" in mode:
            knowledge_prompt += f"""
            TECHNICAL REQUIREMENTS:
            - Domain: {domain}
            - Required Stack: {tech_stack}
            - Strategy: Ask scenario-based questions involving {tech_stack}. Avoid definition questions (e.g., "What is React?"). Instead ask: "How would you optimize a slow React render cycle?"
            """
            
        if "HR" in mode or "Mixed" in mode:
            knowledge_prompt += f"""
            CULTURAL EVALUATION:
            - Focus Areas: {hr_focus}
            - Strategy: Ask behavioral questions. Example: "Tell me about a time you handled a conflict." Look for indicators of {hr_focus}.
            """

        # 5. Final Assembly for Vapi
        system_prompt_for_vapi = f"""
        {persona_prompt}
        
        {behavior_prompt}
        
        CONTEXT FROM JD:
        {job_desc}
        
        {knowledge_prompt}
        
        INTERVIEW GUIDELINES:
        1. Keep responses concise (under 2 sentences).
        2. Wait for the user to finish speaking.
        3. Do not Hallucinate skills the user does not have.
        4. End the interview after gathering sufficient data.
        """

        # 6. Call Groq to structure the output JSON
        # We ask Groq to format this into the JSON structure your UI expects
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an API that generates interview blueprints. Output JSON only."
                },
                {
                    "role": "user",
                    "content": f"""
                    Create a structured interview plan based on this system prompt:
                    {system_prompt_for_vapi}
                    
                    The JSON must contain:
                    1. "system_prompt": The exact text provided above (cleaned up).
                    2. "phases": An array of 3 phases (Intro, Core, Closing).
                    3. "estimated_duration": String.
                    """
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.6,
            response_format={"type": "json_object"},
        )

        response_content = chat_completion.choices[0].message.content
        blueprint = json.loads(response_content)
        
        # Ensure the System Prompt is passed back exactly as we built it (Groq sometimes summarizes it)
        blueprint["system_prompt"] = system_prompt_for_vapi

        return blueprint

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/campaigns/{campaign_id}/save-final")
async def save_final_campaign(campaign_id: str, request: Request):
    """Saves the final Blueprint + Config to MongoDB"""
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401, detail="Not logged in")
    
    try:
        data = await request.json()
        
        # Update the campaign in MongoDB
        db = app.mongodb["campaigns"]
        
        # Ensure we use ObjectId for the query
        from bson import ObjectId
        
        await db.update_one(
            {"_id": ObjectId(campaign_id)}, 
            {"$set": {
                "status": "Ready",
                "blueprint": data.get('blueprint'),
                "config": data.get('config'),
                "updated_at": datetime.utcnow().isoformat()
            }}
        )
        return {"status": "success", "message": "Campaign saved successfully"}
        
    except Exception as e:
        print(f"Save Error: {e}")
        raise HTTPException(status_code=500, detail="Database save failed")


api_key = os.getenv("GROQ_API_KEY")
if api_key:
    print(f"✅ Key found: {api_key[:5]}...") # Prints first 5 chars only
else:
    print("❌ API Key NOT found! Check .env file.")



# --- ADD THIS TO main.py ---

from bson import ObjectId

# 1. Get Specific Campaign Details (Config + Blueprint)
@app.get("/api/campaigns/{campaign_id}")
async def get_campaign_details(campaign_id: str, request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)

    db = app.mongodb["campaigns"]
    campaign = await db.find_one({"_id": ObjectId(campaign_id)})
    
    if campaign:
        campaign["id"] = str(campaign.pop("_id"))
        return campaign
    raise HTTPException(status_code=404, detail="Campaign not found")

# 2. Get Candidates for a Campaign
@app.get("/api/campaigns/{campaign_id}/candidates")
async def get_campaign_candidates(campaign_id: str, request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)

    db = app.mongodb["candidates"]
    cursor = db.find({"campaign_id": campaign_id})
    
    candidates = []
    async for doc in cursor:
        doc["id"] = str(doc.pop("_id"))
        candidates.append(doc)
        
    return {"candidates": candidates}



# --- ADD THIS TO main.py ---

@app.delete("/api/campaigns/{campaign_id}")
async def delete_campaign(campaign_id: str, request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)

    try:
        from bson import ObjectId
        
        # 1. Delete the Campaign
        camp_res = await app.mongodb["campaigns"].delete_one({
            "_id": ObjectId(campaign_id),
            "user_id": user["google_id"] # Security check
        })

        if camp_res.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Campaign not found or access denied")

        # 2. Cleanup: Delete associated candidates
        await app.mongodb["candidates"].delete_many({"campaign_id": campaign_id})

        return {"status": "success", "message": "Campaign deleted"}

    except Exception as e:
        print(f"Delete Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete campaign")




# --- IN main.py ---
# Add this endpoint to fetch Dashboard Stats

@app.get("/api/dashboard-stats")
async def get_dashboard_stats(request: Request):
    user = request.session.get('user')
    if not user:
        return {"error": "Not logged in"}

    # Connect to collections
    candidates_coll = app.mongodb["candidates"]
    campaigns_coll = app.mongodb["campaigns"]
    user_id = user["google_id"]

    # 1. Count Total Candidates
    total_candidates = await candidates_coll.count_documents({"user_id": user_id})

    # 2. Count "Active" (Pending or In Progress)
    active_count = await candidates_coll.count_documents({
        "user_id": user_id, 
        "status": {"$in": ["Pending", "In Progress", "Scheduled"]}
    })

    # 3. Count "Completed" (Selected or Rejected)
    completed_count = await candidates_coll.count_documents({
        "user_id": user_id,
        "status": {"$in": ["Selected", "Rejected"]}
    })

    # 4. Count Total Campaigns
    total_campaigns = await campaigns_coll.count_documents({"user_id": user_id})

    return {
        "total_candidates": total_candidates,
        "active_candidates": active_count,
        "interviews_done": completed_count,
        "total_campaigns": total_campaigns
    }



# --- ADD THESE ENDPOINTS TO THE BOTTOM OF main.py ---

import random

# 1. DASHBOARD STATS (Counts real data from DB)
@app.get("/api/dashboard-stats")
async def get_dashboard_stats(request: Request):
    user = request.session.get('user')
    if not user: return {"error": "Not logged in"}

    user_id = user["google_id"]
    candidates_coll = app.mongodb["candidates"]
    campaigns_coll = app.mongodb["campaigns"]

    # Count actual documents
    total_candidates = await candidates_coll.count_documents({"user_id": user_id})
    
    # Count "Active" (Pending/Scheduled)
    active_count = await candidates_coll.count_documents({
        "user_id": user_id, 
        "status": {"$in": ["Pending", "In Progress", "Scheduled"]}
    })

    # Count "Completed" (Selected/Rejected)
    completed_count = await candidates_coll.count_documents({
        "user_id": user_id,
        "status": {"$in": ["Selected", "Rejected"]}
    })

    return {
        "total_candidates": total_candidates,
        "active_candidates": active_count,
        "interviews_done": completed_count
    }

# 2. LIVE ACTIVITY (Simulates "Active" status for the Dashboard table)
@app.get("/api/dashboard/live-activity")
async def get_live_activity(request: Request):
    user = request.session.get('user')
    if not user: return {"activity": []}

    # Fetch 5 most recent candidates
    cursor = app.mongodb["candidates"].find(
        {"user_id": user["google_id"]}
    ).sort("created_at", -1).limit(5)
    
    candidates = []
    
    # Simulation Logic: Give them "Live" statuses for the UI effect
    demo_statuses = ["Active Session", "Analyzed", "Dialing...", "Scheduled"]
    
    async for doc in cursor:
        doc["id"] = str(doc.pop("_id"))
        
        # If status is default (Pending), assign a random "Live" status
        if doc.get("status", "Pending") == "Pending":
            doc["display_status"] = random.choice(demo_statuses)
            doc["duration"] = "04m 12s" if "Active" in doc["display_status"] else "--"
        else:
            doc["display_status"] = doc.get("status")
            doc["duration"] = doc.get("duration", "--")
            
        candidates.append(doc)
        
    return {"activity": candidates}

# 3. ALL CANDIDATES LIST (For the Candidates Tab)
@app.get("/api/candidates/all")
async def get_all_candidates_list(request: Request):
    user = request.session.get('user')
    if not user: return {"candidates": []}

    cursor = app.mongodb["candidates"].find({"user_id": user["google_id"]}).sort("created_at", -1)
    candidates = []
    
    async for doc in cursor:
        doc["id"] = str(doc.pop("_id"))
        # Ensure a default status exists
        if "status" not in doc: doc["status"] = "Not Contacted"
        candidates.append(doc)
        
    return {"candidates": candidates}







# ==========================================
# 4. CAMPAIGN LAUNCHER (Receives IDs + Prompt)
# ==========================================
class CampaignLaunchRequest(BaseModel):
    campaign_name: str
    vapi_agent_id: str
    vapi_voice_id: str
    system_prompt: str
    candidates: list = []
    strictness: str
    interview_mode: str

@app.post("/api/launch-campaign")
async def launch_campaign(request: CampaignLaunchRequest, req: Request):
    user = req.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    print(f"\n🚀 --- INITIATING CAMPAIGN: {request.campaign_name} ---")
    print(f"👤 Agent ID: {request.vapi_agent_id}")
    print(f"🎙️ Voice ID: {request.vapi_voice_id}")
    print(f"🧠 System Prompt Length: {len(request.system_prompt)} chars")
    print(f"👥 Candidates to Call: {len(request.candidates)}")
    
    # 1. Create Campaign Record in MongoDB
    new_campaign = {
        "user_id": user['google_id'],
        "name": request.campaign_name,
        "status": "Active", # Set to Active immediately
        "created_at": datetime.utcnow(),
        "config": {
            "agent_id": request.vapi_agent_id,
            "voice_id": request.vapi_voice_id,
            "prompt": request.system_prompt,
            "strictness": request.strictness
        },
        "stats": {
            "total": len(request.candidates),
            "completed": 0,
            "pending": len(request.candidates)
        }
    }
    
    result = await app.mongodb["campaigns"].insert_one(new_campaign)
    campaign_id = str(result.inserted_id)

    # 2. Add Candidates to Database (Linked to this Campaign)
    if request.candidates:
        candidate_docs = []
        for cand in request.candidates:
            candidate_docs.append({
                "campaign_id": campaign_id,
                "user_id": user['google_id'],
                "name": cand.get("Name", "Unknown"),
                "email": cand.get("Email", ""),
                "phone": cand.get("Phone", ""),
                "status": "Queued", # Ready to be called
                "created_at": datetime.utcnow()
            })
        
        if candidate_docs:
            await app.mongodb["candidates"].insert_many(candidate_docs)

    # 3. (FUTURE) TRIGGER VAPI CALL HERE
    # In the future, you will loop through 'candidate_docs' here 
    # and call the Vapi API using 'request.vapi_agent_id'
    
    print(f"✅ Campaign {campaign_id} stored in DB.")

    return {
        "status": "success", 
        "campaign_id": campaign_id, 
        "message": "Campaign initialized and candidates queued."
    }


# ==========================================
# 5. DEBUG ENDPOINT (Check what was received)
# ==========================================
@app.get("/api/debug/latest-campaign")
async def get_latest_campaign_debug():
    """
    Returns the raw configuration of the most recently launched campaign.
    Use this to verify if the Frontend sent the correct Agent ID and Prompt.
    """
    # 1. Fetch the last created campaign
    latest_campaign = await app.mongodb["campaigns"].find_one(
        sort=[("created_at", -1)] 
    )

    if not latest_campaign:
        return {"status": "No campaigns found in database."}

    # 2. Return the config for inspection
    return {
        "status": "Found Latest Campaign",
        "campaign_name": latest_campaign.get("name"),
        "received_config": {
            "agent_id": latest_campaign["config"].get("agent_id"),
            "voice_id": latest_campaign["config"].get("voice_id"),
            "strictness": latest_campaign["config"].get("strictness"),
            # Show first 100 chars of prompt to verify it exists
            "system_prompt_preview": latest_campaign["config"].get("prompt", "")[:100] + "..."
        },
        "created_at": latest_campaign.get("created_at")
    }