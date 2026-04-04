# 🚀 Render.com Deployment Guide (Free Tier - 512 MB RAM)

## ⚡ Quick Summary

✅ **512MB RAM will work** with optimizations  
✅ **ResNet18 model fits** (~100 MB)  
✅ **WebSocket works perfectly** on Render  
✅ **Cost: Free** ($0/month)  

---

## 📋 Prerequisites

1. **GitHub Account** - Push code to GitHub
2. **Render Account** - Sign up at https://render.com (free)
3. **Model File** - `terrain_classifier.pth` (~45 MB) in repo
4. **Git** - Installed locally

---

## 🔧 Step 1: Prepare Your Repository

### 1.1 Add `.gitignore` (if needed)
```bash
# Already has model file
terrain_classifier.pth  # Keep this - model is required
```

### 1.2 Verify Files in Root
```
✅ app.py (WebSocket backend)
✅ requirements.txt (dependencies)
✅ terrain_classifier.pth (model - 45MB)
✅ static/ (CSS & JS)
✅ templates/ (HTML)
✅ video.py (frame extraction)
```

### 1.3 Push to GitHub
```bash
git add .
git commit -m "v2.0: WebSocket + video processing + live streaming"
git push origin main
```

---

## 🌐 Step 2: Create Render Service

### 2.1 Go to Render
- Open https://render.com
- Click **"New +"** → **"Web Service"**

### 2.2 Connect GitHub
- Click **"Connect existing repository"**
- Select your `OFFROAD-AI_graphEthon` repo
- Click **"Connect"**

### 2.3 Configure Service

| Setting | Value |
|---------|-------|
| **Name** | offroad-ai |
| **Region** | Oregon (us-west) or closest to you |
| **Branch** | main |
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app --worker-class eventlet -w 1 --threads 2` |
| **Plan** | Free |
| **Environment** | Add secret (see below) |

### 2.4 Add Environment Variable
Click **"Add Environment Variable"**:

```
KEY: FLASK_ENV
VALUE: production
```

Click **"Create Web Service"**

---

## ⚙️ Step 3: Optimize for 512MB RAM

### 3.1 Update `requirements.txt` (already done, but verify)
```
torch==2.0.1
torchvision==0.15.2
flask
flask-socketio
python-socketio
python-engineio
pillow
numpy
opencv-python
gunicorn
eventlet
```

### 3.2 Create `Procfile` (for Render)
```bash
# Create file: Procfile (new file in root)
web: gunicorn app:app --worker-class eventlet -w 1 --threads 2 --timeout 120
```

### 3.3 Optimize `app.py` for RAM
Add this at the top of `app.py` (after imports):

```python
# ================= MEMORY OPTIMIZATION =================

import os

# Reduce PyTorch memory usage
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)

# Disable GPU (not available on free tier anyway)
is_gpu_available = torch.cuda.is_available()
if is_gpu_available:
    # Free up GPU memory if somehow available
    torch.cuda.empty_cache()

# Use CPU only (more RAM-friendly)
device = torch.device("cpu")
print(f"🖥️ Device: {device}")
print(f"💾 Available memory: ~{os.cpu_count()} cores")

# ================= END OPTIMIZATION =================
```

---

## 📤 Step 4: Deploy

### 4.1 Auto-Deploy from GitHub
Once configured, **Render automatically deploys** when you push:

```bash
git push origin main
# Render detects change → builds → deploys
```

### 4.2 Manual Trigger (if needed)
- Go to Render dashboard
- Click your service
- Click **"Manual Deploy"** → **"Deploy latest commit"**

### 4.3 Wait for Build (takes 5-10 minutes)
Watch the logs:
- ✅ "Building"
- ✅ "Installing dependencies"
- ✅ "Deploying"
- ✅ "Live" (green status)

---

## 🌍 Step 5: Access Your App

Once deployed:

```
https://offroad-ai.onrender.com
```

(Replace `offroad-ai` with your actual service name)

---

## 📊 Expected Performance (512 MB RAM)

### Memory Usage
```
Flask Framework: ~30 MB
PyTorch Model: ~100 MB
WebSocket: ~10 MB
Free Space: ~370 MB
```

### Processing Speed
```
Image Classification: 2-4 seconds (CPU)
Video Processing (7 frames): 15-30 seconds
Live Stream: 300-500ms per frame

Frame Rate: 2 FPS recommended
```

### Startup Time
```
Cold start (first request): 15-30 seconds
Subsequent requests: <1 second
```

---

## 🔍 Monitoring & Logs

### View Real-Time Logs
1. Go to Render dashboard
2. Click your service
3. Click **"Logs"** tab
4. Watch live output

### Check Status
- **Green dot** = Running ✅
- **Red/Yellow** = Error or deploying

### Common Logs
```
✅ "Running on http://0.0.0.0:10000"
✅ "Client connected"
✅ "Frame processed"
✅ "Model loaded successfully"
```

---

## ⚠️ Known Limitations (512MB RAM)

### Issue 1: Slow First Request
**Cause**: Model loading takes time  
**Solution**: Already handled with preload in app.py

### Issue 2: Memory Pressure
**Cause**: Processing multiple concurrent users  
**Solution**: Limit concurrent users
- Free tier has 1 worker
- Max ~2-3 concurrent users recommended

### Issue 3: Timeout on Long Operations
**Cause**: Video processing > 30 seconds  
**Solution**: Increase timeout
```python
# In Procfile:
web: gunicorn app:app --worker-class eventlet -w 1 --timeout 300
```

### Issue 4: GPU Not Available
**Expected**: Render free tier has no GPU  
**Solution**: Already using CPU-only (see optimization)

---

## 🛠️ Troubleshooting

### Deploy Shows "Deployment Failed"

**Check logs**:
1. Go to Logs tab
2. Look for error messages
3. Common issues:

```
❌ "No module named 'torch'"
   → Fix: requirements.txt missing packages
   
❌ "Model file not found"
   → Fix: Ensure terrain_classifier.pth in repo
   
❌ "Out of memory"
   → Fix: Reduce model size or worker count
   
❌ "Address already in use"
   → Fix: Only use port from $PORT env variable
```

### Fix Port Configuration

Add to `app.py` (before app.run()):

```python
port = int(os.environ.get('PORT', 10000))
```

Change last line:
```python
# Wrong:
socketio.run(app, host="0.0.0.0", port=10000)

# Correct:
socketio.run(app, host="0.0.0.0", port=port)
```

### Fix Model Loading Error

Ensure model is in GitHub repo:
```bash
# Check file size
ls -lh terrain_classifier.pth

# Should show ~45MB
# If missing, download and add to repo
```

---

## 🚀 Optimization Tips

### Reduce Startup Time
```python
# In app.py - preload models on startup only if needed
def preload_models():
    print("Loading models...")
    load_classifier()
    print("✅ Ready!")

# Call only once
preload_models()
```

### Reduce Per-Request Latency
```python
# Use lighter JPEG compression
def compress_frame(frame_array, quality=50):  # Lower from 60
    # ... rest of function
```

### Reduce Memory Usage
```python
# Limit result history
MAX_RESULTS_HISTORY = 1  # Down from 3
```

---

## 📈 Scaling (If Needed Later)

If 512 MB becomes insufficient:

| Plan | RAM | Cost | Features |
|------|-----|------|----------|
| **Free** | 512 MB | $0/mo | Single instance |
| **Pay-as-you-go** | 1-2 GB | ~$7/mo | Better performance |
| **Pro** | 2+ GB | $12+/mo | Guaranteed uptime |

Simply upgrade in Render dashboard (1 click).

---

## ✅ Deployment Checklist

- [ ] App tested locally with `run.bat` or `./run.sh`
- [ ] All files pushed to GitHub
- [ ] `terrain_classifier.pth` in repository
- [ ] `requirements.txt` has all dependencies
- [ ] `Procfile` created in root
- [ ] `app.py` uses `$PORT` environment variable
- [ ] Memory optimizations added
- [ ] GitHub account connected to Render
- [ ] Service created and configured
- [ ] Build succeeded (green status)
- [ ] Can access `https://your-service.onrender.com`
- [ ] Image upload works
- [ ] Video upload works (7 frames)
- [ ] Live camera works (WebSocket)
- [ ] History displays correctly

---

## 🎯 Testing Deployed App

Once live on Render:

### Test 1: Image Upload
```
1. Go to https://your-service.onrender.com
2. Click "📸 Image" tab
3. Upload a test image
4. Verify result displays
⏱️ Expected: 3-5 seconds
```

### Test 2: Video Upload
```
1. Click "🎬 Video" tab
2. Upload a test video (5-30 seconds)
3. Click "Analyze Video (7 Frames)"
4. Verify 7 frames processed
⏱️ Expected: 20-40 seconds
```

### Test 3: Live Stream
```
1. Click "📹 Live Stream" tab
2. Click "▶️ Start Stream"
3. Allow camera access
4. Verify real-time results
⏱️ Expected: 300-500ms per frame
```

### Test 4: Connection Indicator
```
Check status icon in footer:
✅ 🟢 Connected - All good
❌ 🔴 Disconnected - Reload page
```

---

## 📞 Support & Resources

### Render Documentation
- https://render.com/docs
- https://render.com/docs/deploy-flask
- https://render.com/docs/web-services

### Common Issues
- **Can't connect**: Check Render status page
- **Slow performance**: Monitor logs, check CPU usage
- **Model not loading**: Verify file size and permissions
- **WebSocket fails**: Ensure `eventlet` worker class

---

## 🎓 What Will Happen

### Deployment Process
```
1. You push to GitHub (git push origin main)
   ↓
2. Render detects change in 10-30 seconds
   ↓
3. Starts build (Building status)
   ↓
4. Installs dependencies (5-10 minutes)
   ↓
5. Downloads model file (~45 MB - 1-2 minutes)
   ↓
6. Starts server (starts Flask-SocketIO)
   ↓
7. Preloads ResNet18 model (10-30 seconds)
   ↓
8. Service goes live (green status)
   ↓
9. Your app is accessible on the internet! 🎉
```

### First User
```
User visits https://offroad-ai.onrender.com
   ↓
App loads HTML/CSS/JS (fast)
   ↓
Browser connects WebSocket
   ↓
User uploads image or starts live stream
   ↓
GPU not used (CPU processes)
   ↓
Result returns in 2-5 seconds
   ↓
UI updates in real-time
```

---

## 💡 Cost Breakdown

```
Render Free Tier:
✅ 512 MB RAM
✅ Shared CPU
✅ Web: $0/month
✅ 750 hours/month free

Total Cost: $0 🎉
```

---

## 🔐 Security Notes

Your app will be **public**:
- Anyone can access it
- Use responsibly
- Consider adding API keys later if needed

---

## 🎉 Ready to Deploy!

1. **Optimize app.py** (add memory optimizations)
2. **Create Procfile** in root directory
3. **Push to GitHub**
4. **Create Render service**
5. **Watch logs** during build
6. **Test all features** once live
7. **Share link** with others!

---

**Estimated Total Time**: 30-45 minutes

**Questions?** Check Render docs or review logs for specific errors.

---

**Version**: 2.0  
**Status**: Ready for Render Free Tier  
**Last Updated**: April 2024
