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
  const frames = await extractVideoFrames(videoFile)

  let decisions = []
  let html = ""

  for (let i = 0; i < frames.length; i++) {
    let fd = new FormData()
    fd.append("file", frames[i])

    let res = await fetch("/predict-image", {
      method: "POST",
      body: fd
    })

    let data = await res.json()

    decisions.push(data.decision)

    html += `
      <div>
        <img src="data:image/png;base64,${data.mask}" width="100%">
        <p>Terrain: ${data.terrain}</p>
        <p>Decision: ${data.decision}</p>
      </div>
    `
  }

  document.getElementById("videoFramesResults").innerHTML = html
  document.getElementById("videoFinalDecision").innerHTML =
    "Final: " + decisions[0]
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
    frameCount++

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