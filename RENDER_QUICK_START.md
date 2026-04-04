# 🚀 RENDER.COM DEPLOYMENT - Quick Start (5 Minutes)

## ✅ Pre-Deployment Checklist (Run Locally First)

```bash
# 1. Test locally
run.bat  # Windows
# or
./run.sh  # Linux/Mac

# Should show:
# ✅ Models loaded
# ✅ Running on http://127.0.0.1:10000
# ✅ "Client connected" when you visit in browser
```

---

## 📋 Files Ready for Deployment

```
✅ app.py - Optimized for 512MB RAM (CPU-only)
✅ Procfile - Render launch configuration
✅ requirements.txt - Updated with eventlet
✅ terrain_classifier.pth - Model file (45MB)
✅ templates/index.html - UI
✅ static/script.js - WebSocket client
✅ static/style.css - Styling
✅ video.py - 7-frame extraction
```

---

## 🔑 Step-by-Step Deployment

### Step 1: Push to GitHub (2 minutes)

```bash
cd c:\Nishchay\OFFROAD-AI_graphEthon

git add -A

git commit -m "v2.0: WebSocket + video + Render deployment"

git push origin main
```

**Expected output**:
```
Counting objects: 50, done.
Compressing objects: 100% (30/30), done.
Writing objects: 100% (50/50), 200 KB
Remote: Compressing source files... done.
✅ Success
```

### Step 2: Create Render Account (1 minute)

1. Go to **https://render.com**
2. Click **"Sign Up"** (use GitHub account - easier)
3. Authorize GitHub connection
4. Click **"Authorize render-rnw"**

### Step 3: Create Web Service (2 minutes)

1. Click **"New +"** (top right)
2. Select **"Web Service"**
3. Click **"Connect existing repository"**
4. Search for **"OFFROAD-AI_graphEthon"**
5. Click **"Connect"**

### Step 4: Configure Service (2 minutes)

Fill in these fields:

| Field | Value |
|-------|-------|
| **Name** | `offroad-ai` |
| **Region** | Select closest (e.g., Ohio, Oregon) |
| **Branch** | `main` |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app --worker-class eventlet -w 1 --threads 2` |
| **Plan** | `Free` |

### Step 5: Create & Deploy

1. Click **"Create Web Service"**
2. Watch build logs (takes 3-5 minutes)

**Expected build sequence**:
```
🔨 Building...
📦 Installing dependencies (pip install)
🔽 Downloading torch (~800 MB - takes 2-3 min)
🔽 Downloading model file (.pth - 45 MB)
✅ Build successful
🚀 Deploying...
✅ Live (green status)
```

---

## 🎉 Your App is Live!

Once status shows **🟢 Live**, your app is at:

```
https://offroad-ai.onrender.com
```

Click the link and test:

1. ✅ **Image Tab** - Upload image
2. ✅ **Video Tab** - Upload video (processes 7 frames)
3. ✅ **Live Stream** - Camera streaming
4. ✅ **History Tab** - Previous results

---

## 📊 Expected Performance

| Metric | Performance |
|--------|-------------|
| **Cold Start** | 15-30 sec (first request) |
| **Image Analysis** | 3-5 seconds |
| **Video (7 frames)** | 20-40 seconds |
| **Live Stream FPS** | 2-3 FPS |
| **Concurrent Users** | 2-3 max |
| **RAM Usage** | ~250 MB / 512 MB available |
| **Cost** | **$0/month** 🎉 |

---

## 🔍 Monitor Your App

### View Real-Time Logs
1. Go to **https://dashboard.render.com**
2. Click **"offroad-ai"** service
3. Click **"Logs"** tab
4. Watch live output

### Expected Logs
```
✅ "Device: CPU (Render Free Tier)"
✅ "PRELOADING MODELS AT STARTUP..."
✅ "Classifier loaded successfully"
✅ "Running on http://0.0.0.0:10000"
✅ "Client connected: [session-id]"
✅ "Processing frame..."
```

### Check Status
- **🟢 Live** = Running perfectly
- **🟡 Spinning** = Building or deploying
- **🔴 Dead** = Error (check logs)

---

## ⚠️ Common Issues & Fixes

### "Build Failed" Error

**Check logs** → Look for error:

```
❌ "Could not find a version that satisfies the requirement"
   → Fix: Requirements syntax issue
   → Solution: Verify requirements.txt format

❌ "No space left on device"
   → Fix: Package too large
   → Solution: Remove unnecessary packages

❌ "File not found: terrain_classifier.pth"
   → Fix: Model file missing from repo
   → Solution: Verify file exists in GitHub
```

### App Runs but Data Processing is Slow

**Expected behavior** (first request slower):
```
First request: 20-30 seconds (models loading)
Subsequent: 3-5 seconds (cached)
This is normal on free tier ✅
```

**Optimize if needed**:
```python
# In app.py, reduce history size:
MAX_RESULTS_HISTORY = 1  # from 3
```

### WebSocket Connection Fails

**Symptom**: Status shows 🔴 Disconnected

**Fixes**:
1. Refresh page (hard refresh: Ctrl+Shift+R)
2. Wait 30 seconds (cold start can be slow)
3. Check browser console (F12 → Console)
4. Check Render logs for errors

### Model Takes Forever to Load

**Normal behavior**:
```
First deploy: ~30-60 seconds (downloading + loading)
Subsequent: ~5-10 seconds (model cached)
```

**This is expected on free tier** - models are large (~100 MB).

---

## 🎓 Render Free Tier Details

### What You Get
```
✅ 512 MB RAM
✅ Shared CPU (1 vCPU equivalent)
✅ 750 FREE hours/month (~31 days continuous)
✅ WebSocket support
✅ Custom domains
✅ SSL certificates (free)
✅ Auto-redeploy on git push
```

### Limitations
```
⚠️ Spins down after 15 min inactivity
⚠️ Cold start takes 10-15 seconds
⚠️ Shared resources (slow at times)
⚠️ Max 2-3 concurrent users recommended
```

### When to Upgrade
```
If you get:
🔴 Memory errors
🔴 Frequent timeouts
🔴 >10 concurrent users

Upgrade to Pay-as-you-go (~$7/month)
```

---

## 📱 Share Your Live App

Once deployed, you can share the link:

```
My OFFROAD AI is live! 🚗
Check it out: https://offroad-ai.onrender.com
```

Anyone with the link can:
- ✅ Upload images
- ✅ Upload videos
- ✅ Use live camera stream
- ✅ View results

---

## 🎯 What Happens Next

### Auto-Updates
```
Every time you push to GitHub:
git push origin main
     ↓
Render detects change (within 30 seconds)
     ↓
Automatically rebuilds (5-10 minutes)
     ↓
Auto-deploys new version
     ↓
Zero downtime if using multiple instances
```

### Or Manual Update
In Render dashboard:
1. Click service
2. Click **"Manual Deploy"**
3. Click **"Deploy latest commit"**
4. Done! (5-10 min)

---

## 🔐 Security & Environment

### Database/Secrets
Not needed for this app - stateless.

But if you ever need secrets:
```
1. Go to Render service
2. Click "Environment"
3. Add secret key-value pairs
4. Auto-deploy with secrets
5. Access in Python: os.environ.get('KEY')
```

### Logs Privacy
- Logs are private to your account
- Users can't see backend logs
- Only see public responses

---

## 💰 Cost Summary

```
Render Free Tier:

Web Service: $0/month
  - 512 MB RAM ✅
  - Shared CPU ✅
  - 750 hours/month ✅

Storage: $0/month
  - Included ✅

Total: $0/month 🎉

Upgrade to Pay-as-you-go:
  - ~$7-12/month for better performance
  - Only if needed
```

---

## ✅ Full Checklist

Before deploying:
- [ ] App works locally (`run.bat` successful)
- [ ] All files pushed to GitHub
- [ ] Procfile in root directory
- [ ] requirements.txt has eventlet
- [ ] terrain_classifier.pth in repo

During deployment:
- [ ] Build succeeded (green status)
- [ ] Logs show "Live" status
- [ ] Can access the URL

After live:
- [ ] Image upload works
- [ ] Video upload works
- [ ] Live stream works
- [ ] Results display correctly

---

## 🆘 Still Having Issues?

**Resources**:
1. Check **RENDER_DEPLOYMENT.md** (detailed guide)
2. Review **Render logs** (Render dashboard → Logs)
3. Check **Browser console** (F12 → Console)
4. Review **app.py imports** (verify all packages installed)

---

## 🎉 Success!

Your OFFROAD AI is now running on **Render's free tier**:

✅ Deployed  
✅ Live on internet  
✅ Fast WebSocket  
✅ Free forever (or upgrade when needed)  
✅ Auto-updates from GitHub  
✅ 512 MB RAM available  

**Share your link**: `https://offroad-ai.onrender.com`

---

**Deployment Time**: ~15 minutes total  
**Cost**: $0/month  
**Status**: 🟢 Production Ready  
**Last Updated**: April 2024
