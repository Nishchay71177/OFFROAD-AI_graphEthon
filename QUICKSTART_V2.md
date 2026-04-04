# 🚀 QUICK START - OFFROAD AI v2.0

## ⚡ Get Running in 30 Seconds

### Windows Users
```bash
run.bat
```

### Linux/Mac Users
```bash
chmod +x run.sh
./run.sh
```

### Manual Setup
```bash
pip install -r requirements.txt
python app.py
```

## 🌐 Access the App
Open your browser and go to:
```
http://localhost:10000
```

---

## 📸 What You Can Do

### 1. Upload & Analyze Image
- Click **📸 Image** tab
- Drag & drop or click to upload terrain image
- Click **Analyze Image**
- See result with segmentation mask

### 2. Live Stream from Camera
- Click **📹 Live Stream** tab
- Click **▶️ Start Stream**
- Allow camera access (browser will ask)
- Watch real-time analysis (terrain + decision + mask)
- Results update every ~300ms (~3 FPS)

### 3. View History
- Click **📊 History** tab
- See last 3 analyzed results
- Each result shows:
  - Terrain classification (Easy/Moderate/Rough/Very Rough)
  - Confidence percentage
  - Navigation decision
  - Timestamp
  - Segmentation mask

---

## 🎯 Terrain Classes & Decisions

### Terrain Types
| Type | Color | Meaning |
|------|-------|---------|
| **Easy** | 🟢 Green | Safe to proceed |
| **Moderate** | 🟡 Yellow | Go slow |
| **Rough** | 🟠 Orange | Navigate carefully |
| **Very Rough** | 🔴 Red | Stop/Danger |

### Navigation Decisions
| Decision | Meaning |
|----------|---------|
| **GO STRAIGHT** | Clear path ahead |
| **GO LEFT/RIGHT** | Better path available |
| **GO SLOW STRAIGHT/LEFT/RIGHT** | Proceed with caution |
| **TURN LEFT/RIGHT** | Many obstacles detected |
| **STOP** | Terrain too dangerous |

---

## 🔌 Connection Status

- **🟢 Connected** = Ready to process frames
- **🔴 Disconnected** = Waiting for server
- **⏳ Connecting** = Establishing connection

If disconnected:
1. Check if server is running (`run.bat` or `python app.py`)
2. Refresh browser (F5)
3. Check server logs for errors

---

## ⚙️ Settings to Customize

### Adjust Frame Rate
**File**: `static/script.js` (Line ~250)

```javascript
// Current: 300ms = ~3 FPS
setTimeout(captureAndProcessFrame, 300);

// Try these:
// 200ms = 5 FPS (faster, more CPU)
// 500ms = 2 FPS (slower, less CPU)
// 1000ms = 1 FPS (minimum)
```

### Adjust Image Quality
**File**: `static/script.js` (Line ~240)

```javascript
// Current: 70% quality
canvas.toDataURL('image/jpeg', 0.7);

// Try these:
// 0.5 = 50% (smallest, fastest)
// 0.9 = 90% (larger, better quality)
```

### Adjust History Size
**File**: `app.py` (Line ~33)

```python
# Current: 3 results
MAX_RESULTS_HISTORY = 3

# Try these:
# 1 = Only current result
# 5 = Last 5 results
# 10 = Last 10 results
```

---

## 🐛 Troubleshooting

### "Cannot access camera"
- ✅ Grant browser camera permission
- ✅ Check Windows Privacy Settings → Camera
- ✅ Try different browser (Chrome recommended)

### "Server not responding"
- ✅ Check if `run.bat` or `python app.py` is running
- ✅ Verify port 10000 is not in use
- ✅ Check server console for error messages

### "WebSocket connection failed"
- ✅ Refresh browser (F5)
- ✅ Check browser console (F12 → Console tab)
- ✅ Restart server and browser

### "Model fails to load"
- ✅ Ensure `terrain_classifier.pth` exists in project root
- ✅ File should be ~45 MB
- ✅ Verify file is not corrupted

### "Slow/Laggy results"
- ✅ Reduce frame rate (increase milliseconds)
- ✅ Reduce image quality (lower percentage)
- ✅ Close other browser tabs
- ✅ Check CPU/GPU usage

---

## 📊 Monitor Performance

### In Browser Console (F12)
```
✅ Connected to server
Processing frame 1...
Processing frame 2...
...
```

### In Server Terminal
```
✅ Client connected: [ID]
Processing terrain classification...
✅ Frame result sent to client
```

### Metrics to Watch
- **FPS Counter**: Should be 2-3 (in top right of Live tab)
- **Frame Counter**: Should keep incrementing
- **Connection Status**: Should be 🟢 Connected
- **Processing Time**: <500ms is good

---

## 📱 Mobile Access

Works on mobile devices with built-in cameras!

1. Open `http://<your-computer-ip>:10000` on mobile
2. Grant camera permission
3. Use Live Stream tab
4. See real-time terrain analysis

**Note**: Performance may be slower on mobile due to less powerful hardware.

---

## 🌐 Deployment to Render.com

### Prerequisites
- GitHub account with repo
- Render.com account (free tier available)
- Model file uploaded to repo (or fetched dynamically)

### Steps
1. Push code to GitHub
2. Connect Render.com to GitHub
3. Create new Web Service
4. Set Build Command: `pip install -r requirements.txt`
5. Set Start Command: `gunicorn app:app`
6. Deploy!

### Common Issues
- ❌ Model file too large → Use Git LFS
- ❌ Port conflicts → Use `$PORT` environment variable
- ❌ WebSocket issues → Render supports it ✅

See [DEPLOYMENT_V2.md](DEPLOYMENT_V2.md) for detailed guide.

---

## 📚 More Information

- **Full Guide**: See [DEPLOYMENT_V2.md](DEPLOYMENT_V2.md)
- **Implementation Details**: See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Technical Docs**: See [README.md](README.md)

---

## ✨ Key Features

✅ **Real-time WebSocket Streaming**  
✅ **Frame Compression (95% smaller)**  
✅ **Last 3 Results History**  
✅ **Mobile Responsive Design**  
✅ **GPU/CPU Support**  
✅ **No Video Buffering Issues**  
✅ **Modern UI with Glassmorphic Design**  
✅ **Works on Render.com**  

---

## 🎓 How It Works

```
You → Start Stream
  ↓
Browser Captures Frame (every 300ms)
  ↓
Compresses to JPEG (70% quality)
  ↓
Sends via WebSocket
  ↓
Server Processes:
  - ResNet18 classifies terrain
  - Edge detection analyzes obstacles
  - Generates navigation decision
  ↓
Returns Result via WebSocket
  ↓
Browser Displays:
  - Terrain (color badge)
  - Confidence %
  - Decision + emoji
  - Segmentation mask
  ↓
Updates History (last 3)
  ↓
Repeat! 🔄
```

---

**Version**: 2.0  
**Status**: ✅ Production Ready  
**Last Updated**: April 2024
