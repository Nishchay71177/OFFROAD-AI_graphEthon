let imageFile
let videoFile

// ============= TAB SWITCHING =============

function switchTab(tabName) {
  const tabs = document.querySelectorAll('.tab-content')
  const buttons = document.querySelectorAll('.tab-btn')
  
  tabs.forEach(tab => tab.classList.remove('active'))
  buttons.forEach(btn => btn.classList.remove('active'))
  
  document.getElementById(tabName + '-tab').classList.add('active')
  event.target.classList.add('active')
}

// ============= IMAGE UPLOAD =============

const dropZone = document.getElementById("dropZone")
const input = document.getElementById("fileInput")

dropZone.onclick = () => input.click()

input.onchange = e => {
  imageFile = e.target.files[0]
  previewImage(imageFile)
}

dropZone.ondragover = e => {
  e.preventDefault()
  dropZone.style.background = "rgba(56,189,248,0.15)"
}

dropZone.ondragleave = () => {
  dropZone.style.background = ""
}

dropZone.ondrop = e => {
  e.preventDefault()
  dropZone.style.background = ""
  imageFile = e.dataTransfer.files[0]
  previewImage(imageFile)
}

function previewImage(file) {
  let reader = new FileReader()
  reader.onload = e => {
    document.getElementById("preview").src = e.target.result
  }
  reader.readAsDataURL(file)
}

async function predictImage() {
  let formData = new FormData()
  formData.append("file", imageFile)

  document.getElementById("decision").innerHTML = "Analyzing..."

  let res = await fetch("/predict-image", {
    method: "POST",
    body: formData
  })

  let data = await res.json()

  document.getElementById("terrain").innerHTML =
    "Terrain: " + data.terrain

  document.getElementById("decision").innerHTML =
    "Navigation: " + data.decision

  document.getElementById("maskPreview").src =
    "data:image/png;base64," + data.mask
}

// ============= VIDEO UPLOAD =============

const dropZoneVideo = document.getElementById("dropZoneVideo")
const inputVideo = document.getElementById("fileInputVideo")

dropZoneVideo.onclick = () => inputVideo.click()

inputVideo.onchange = e => {
  videoFile = e.target.files[0]
  previewVideoInfo(videoFile)
}

dropZoneVideo.ondragover = e => {
  e.preventDefault()
  dropZoneVideo.style.background = "rgba(56,189,248,0.15)"
}

dropZoneVideo.ondragleave = () => {
  dropZoneVideo.style.background = ""
}

dropZoneVideo.ondrop = e => {
  e.preventDefault()
  dropZoneVideo.style.background = ""
  videoFile = e.dataTransfer.files[0]
  previewVideoInfo(videoFile)
}

function previewVideoInfo(file) {
  const sizeMB = (file.size / (1024 * 1024)).toFixed(2)
  document.getElementById("videoInfo").innerHTML = 
    `<strong>Video File:</strong> ${file.name}<br><strong>Size:</strong> ${sizeMB} MB`
}

// Extract frames
function extractVideoFrames(videoFile) {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video')
    video.src = URL.createObjectURL(videoFile)

    video.onloadedmetadata = () => {
      const frames = []
      let i = 0

      function grab() {
        if (i >= 5) return resolve(frames)

        video.currentTime = (video.duration / 5) * i
      }

      video.onseeked = () => {
        const canvas = document.createElement('canvas')
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        canvas.getContext('2d').drawImage(video, 0, 0)

        canvas.toBlob(blob => {
          frames.push(blob)
          i++
          grab()
        })
      }

      grab()
    }
  })
}

async function predictVideo() {
  if (!videoFile) {
    alert("Please select a video file first")
    return
  }

  let formData = new FormData()
  formData.append("file", videoFile)

  document.getElementById("videoResultsContainer").style.display = "block"
  document.getElementById("videoFramesResults").innerHTML = "<p style='text-align:center; padding:20px;'>Processing video...</p>"
  document.getElementById("videoFinalDecision").innerHTML = ""

  try {
    const res = await fetch("/predict-video", {
      method: "POST",
      body: formData
    })

    const data = await res.json()

    if (!data.success) {
      alert("Error: " + data.error)
      return
    }

    // Display analysis results
    let html = ""
    if (data.frame_predictions && data.frame_predictions.length > 0) {
      data.frame_predictions.forEach((frame, index) => {
        html += `
          <div style="border: 1px solid rgba(56,189,248,0.3); border-radius: 8px; padding: 15px; background: rgba(30,41,59,0.6);">
            <h4 style="margin-top: 0; margin-bottom: 10px; color: #38bdf8;">Sample ${index + 1}</h4>
            <p style="margin: 8px 0; font-size: 13px;"><strong>Terrain Type:</strong> ${frame.terrain}</p>
            <p style="margin: 8px 0; font-size: 13px;"><strong>Recommendation:</strong> ${frame.decision}</p>
            <p style="margin: 8px 0; font-size: 12px; color: #94a3b8; font-style: italic;">${frame.decision_description}</p>
            <div style="margin-top: 10px;">
              <p style="margin: 5px 0; font-size: 11px; color: #64748b;">Segmentation Mask:</p>
              <img src="data:image/png;base64,${frame.mask}" style="width: 100%; border-radius: 6px; border: 1px solid rgba(56,189,248,0.2);">
            </div>
          </div>
        `
      })
    }

    document.getElementById("videoFramesResults").innerHTML = html

    // Display final verdict
    document.getElementById("videoFinalDecision").innerHTML = `
      <div style="background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(56,189,248,0.1)); padding: 20px; border-radius: 8px; border-left: 4px solid #38bdf8; margin-top: 20px;">
        <h3 style="margin-top: 0; margin-bottom: 15px; color: #38bdf8;">Video Analysis Summary</h3>
        <p style="margin: 8px 0; font-size: 13px;">Most Critical Terrain: <strong>${data.final_terrain}</strong></p>
        <div style="background: rgba(34,197,94,0.1); padding: 12px; border-radius: 6px; margin-top: 12px; border: 1px solid rgba(34,197,94,0.3);">
          <p style="margin: 5px 0; font-size: 14px; font-weight: bold; color: #22c55e;">FINAL VERDICT: ${data.final_decision}</p>
          <p style="margin: 8px 0; font-size: 12px; color: #86efac;">${data.final_decision_description}</p>
        </div>
      </div>
    `
  } catch (error) {
    alert("Error processing video: " + error.message)
  }
}

// ============= LIVE =============

let liveStream = null
let analysisRunning = false
let frameCount = 0

async function startLiveAnalysis() {
  try {
    liveStream = await navigator.mediaDevices.getUserMedia({ 
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false 
    })
    
    const video = document.getElementById("liveVideo")
    video.srcObject = liveStream

    document.getElementById("liveVideoContainer").style.display = "block"
    document.getElementById("liveResultsContainer").style.display = "block"
    document.getElementById("startBtn").style.display = "none"
    document.getElementById("stopBtn").style.display = "inline-block"

    analysisRunning = true
    frameCount = 0
    console.log("Live analysis started")
    analysisLoop()
  } catch (error) {
    alert("Camera access denied: " + error.message)
  }
}

function stopLiveAnalysis() {
  analysisRunning = false
  if (liveStream) {
    liveStream.getTracks().forEach(t => t.stop())
  }
  
  document.getElementById("liveVideoContainer").style.display = "none"
  document.getElementById("liveResultsContainer").style.display = "none"
  document.getElementById("startBtn").style.display = "inline-block"
  document.getElementById("stopBtn").style.display = "none"
  
  document.getElementById("liveTerrainType").innerHTML = ""
  document.getElementById("liveDecision").innerHTML = ""
  document.getElementById("liveMaskCapture").src = ""
  
  console.log("Live analysis stopped")
}

async function analysisLoop() {
  if (!analysisRunning) return

  try {
    const video = document.getElementById("liveVideo")

    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0)

    // Show analyzing status
    document.getElementById("liveTerrainType").innerHTML = "Analyzing..."
    document.getElementById("liveDecision").innerHTML = ""

    const blob = await new Promise(resolve => {
      canvas.toBlob(resolve, 'image/jpeg', 0.85)
    })

    const fd = new FormData()
    fd.append("file", blob)

    const res = await fetch("/predict-image", {
      method: "POST",
      body: fd
    })

    if (!res.ok) throw new Error("API error")

    const data = await res.json()

    console.log(`Frame ${frameCount}: ${data.terrain} - ${data.decision}`)

    // Display results immediately
    document.getElementById("liveTerrainType").innerHTML = `Terrain: ${data.terrain}`
    document.getElementById("liveDecision").innerHTML = `Decision: ${data.decision}`
    document.getElementById("liveMaskCapture").src = "data:image/png;base64," + data.mask
    document.getElementById("liveFrameCapture").src = canvas.toDataURL('image/jpeg')

  } catch (error) {
    console.error("Error:", error)
    document.getElementById("liveTerrainType").innerHTML = "Error: " + error.message
  }

  // Continue loop - 1.5s between frames
  if (analysisRunning) {
    setTimeout(analysisLoop, 1500)
  }
}