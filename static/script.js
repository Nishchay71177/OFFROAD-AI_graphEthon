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

dropZone.ondragleave = e => {
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

  document.getElementById("confidence").innerHTML =
    "Confidence: " + data.confidence + "%"

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

dropZoneVideo.ondragleave = e => {
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

// Extract 5 frames from video
function extractVideoFrames(videoFile) {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video')
    video.crossOrigin = 'anonymous'
    video.src = URL.createObjectURL(videoFile)
    
    video.addEventListener('loadedmetadata', () => {
      const duration = video.duration
      const frameCount = 5
      const frames = []
      let currentFrameIndex = 0
      
      function extractFrame() {
        if (currentFrameIndex >= frameCount) {
          resolve(frames)
          return
        }
        
        const time = (duration / frameCount) * currentFrameIndex
        video.currentTime = time
      }
      
      video.addEventListener('seeked', () => {
        try {
          const canvas = document.createElement('canvas')
          canvas.width = video.videoWidth
          canvas.height = video.videoHeight
          const ctx = canvas.getContext('2d')
          ctx.drawImage(video, 0, 0)
          
          canvas.toBlob(blob => {
            frames.push(blob)
            currentFrameIndex++
            
            if (currentFrameIndex < frameCount) {
              extractFrame()
            } else {
              resolve(frames)
            }
          }, 'image/jpeg', 0.9)
        } catch (error) {
          reject(error)
        }
      }, { once: false })
      
      extractFrame()
    }, { once: true })
    
    video.addEventListener('error', (e) => {
      reject(new Error('Video load error: ' + e.message))
    }, { once: true })
  })
}

async function predictVideo() {
  if (!videoFile) {
    alert("Please select a video file first")
    return
  }

  document.getElementById("videoFinalDecision").innerHTML = "Extracting frames..."
  document.getElementById("videoDecisions").innerHTML = ""
  document.getElementById("videoResultsContainer").style.display = "none"

  try {
    console.log("Starting frame extraction...")
    
    // Extract 5 frames
    const frames = await extractVideoFrames(videoFile)
    console.log("Frames extracted:", frames.length)
    
    if (frames.length === 0) {
      throw new Error("No frames could be extracted from the video")
    }
    
    const results = []
    const decisions = []
    
    document.getElementById("videoFinalDecision").innerHTML = "Analyzing frames..."
    
    // Analyze each frame
    for (let i = 0; i < frames.length; i++) {
      console.log(`Analyzing frame ${i + 1}/${frames.length}...`)
      
      const formData = new FormData()
      formData.append("file", frames[i], `frame_${i}.jpg`)
      
      try {
        const res = await fetch("/predict-image", {
          method: "POST",
          body: formData
        })
        
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`)
        }
        
        const data = await res.json()
        console.log(`Frame ${i} result:`, data)
        
        results.push({
          index: i,
          terrain: data.terrain,
          confidence: data.confidence,
          decision: data.decision,
          mask: data.mask
        })
        decisions.push(data.decision)
      } catch (frameError) {
        console.error(`Error processing frame ${i}:`, frameError)
        throw frameError
      }
    }
    
    console.log("All frames processed. Results:", results)
    
    // Display results
    const resultsHTML = results.map((r, idx) => `
      <div style="border:1px solid rgba(255,255,255,0.2); padding:15px; border-radius:10px;">
        <h4>Frame ${idx + 1}</h4>
        <img src="data:image/png;base64,${r.mask}" style="width:100%; border-radius:8px; margin:10px 0;">
        <p><strong>Terrain:</strong> ${r.terrain}</p>
        <p><strong>Confidence:</strong> ${r.confidence}%</p>
        <p><strong>Decision:</strong> ${r.decision}</p>
      </div>
    `).join('')
    
    document.getElementById("videoFramesResults").innerHTML = resultsHTML
    document.getElementById("videoResultsContainer").style.display = "block"
    
    // Display summary
    document.getElementById("videoDecisions").innerHTML =
      "<strong>Frame Decisions:</strong> " + decisions.join(" → ")

    const finalDecision = decisions.reduce((a, b, _, arr) => 
      arr.filter(x => x === a).length > arr.filter(x => x === b).length ? a : b
    )
    
    document.getElementById("videoFinalDecision").innerHTML =
      "<strong>Overall Decision:</strong> " + finalDecision
      
    console.log("Analysis complete!")
      
  } catch (error) {
    console.error("Error analyzing video:", error)
    document.getElementById("videoFinalDecision").innerHTML = 
      "❌ Error: " + error.message + ". Check browser console (F12) for details."
  }
}

// ============= LIVE VIDEO ANALYSIS =============

let liveStream = null
let analysisRunning = false
let frameCount = 0

async function startLiveAnalysis() {
  try {
    // Check if browser supports camera access
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Your browser does not support camera access. Use HTTPS or localhost instead.")
    }
    
    console.log("Requesting camera access...")
    
    // Show permission request alert
    const userConfirmed = confirm("📷 Allow camera access?\n\nYour browser will request permission to use your webcam for live terrain analysis.")
    
    if (!userConfirmed) {
      console.log("Camera access denied by user")
      return
    }
    
    // Request camera access
    liveStream = await navigator.mediaDevices.getUserMedia({
      video: { 
        width: { ideal: 640 }, 
        height: { ideal: 480 }
      },
      audio: false
    })
    
    console.log("Camera access granted!")
    
    // Display video element
    const videoElement = document.getElementById("liveVideo")
    videoElement.srcObject = liveStream
    
    // Wait for video to load
    await new Promise(resolve => {
      videoElement.onloadedmetadata = resolve
    })
    
    document.getElementById("liveVideoContainer").style.display = "block"
    document.getElementById("liveResultsContainer").style.display = "block"
    
    // Update button visibility
    document.getElementById("startBtn").style.display = "none"
    document.getElementById("stopBtn").style.display = "inline-block"
    
    // Start analysis loop
    analysisRunning = true
    frameCount = 0
    console.log("Starting live analysis loop...")
    analysisLoop()
    
  } catch (error) {
    console.error("Camera access error:", error)
    
    let errorMessage = "❌ Cannot access camera.\n\n"
    
    if (error.message.includes("HTTPS") || error.message.includes("localhost")) {
      errorMessage += "⚠️ HTTPS Required!\n\n"
      errorMessage += "Camera access requires:\n"
      errorMessage += "• HTTPS connection (secure), OR\n"
      errorMessage += "• localhost/127.0.0.1\n\n"
      errorMessage += "Current: " + window.location.href
    } else if (error.name === 'NotAllowedError') {
      errorMessage += "Camera permission was denied.\n\n"
      errorMessage += "Enable camera access in browser settings."
    } else if (error.name === 'NotFoundError') {
      errorMessage += "No camera device found on this computer."
    } else if (error.name === 'NotReadableError') {
      errorMessage += "Camera is already in use by another application."
    } else {
      errorMessage += error.message
    }
    
    alert(errorMessage)
  }
}

function stopLiveAnalysis() {
  console.log("Stopping live analysis...")
  
  // Stop the stream
  analysisRunning = false
  if (liveStream) {
    liveStream.getTracks().forEach(track => track.stop())
    liveStream = null
  }
  
  // Hide video element
  document.getElementById("liveVideoContainer").style.display = "none"
  document.getElementById("liveResultsContainer").style.display = "none"
  
  // Update button visibility
  document.getElementById("startBtn").style.display = "inline-block"
  document.getElementById("stopBtn").style.display = "none"
  
  // Clear results
  document.getElementById("liveTerrainType").innerHTML = ""
  document.getElementById("liveConfidence").innerHTML = ""
  document.getElementById("liveDecision").innerHTML = ""
  document.getElementById("liveFrameCount").innerHTML = ""
}

async function analysisLoop() {
  if (!analysisRunning) return
  
  try {
    const videoElement = document.getElementById("liveVideo")
    
    // Capture frame from video
    const canvas = document.createElement('canvas')
    canvas.width = videoElement.videoWidth
    canvas.height = videoElement.videoHeight
    
    const ctx = canvas.getContext('2d')
    ctx.drawImage(videoElement, 0, 0)
    
    // Convert canvas to blob
    canvas.toBlob(async (blob) => {
      try {
        // Create form data and send to backend
        const formData = new FormData()
        formData.append("file", blob, "frame.jpg")
        
        const res = await fetch("/predict-image", {
          method: "POST",
          body: formData
        })
        
        if (!res.ok) throw new Error("Analysis failed")
        
        const data = await res.json()
        frameCount++
        
        console.log(`Live Frame ${frameCount}:`, data)
        
        // Display results in real-time
        document.getElementById("liveTerrainType").innerHTML = 
          `<strong>Terrain:</strong> ${data.terrain}`
        
        document.getElementById("liveConfidence").innerHTML = 
          `<strong>Confidence:</strong> ${data.confidence}%`
        
        document.getElementById("liveDecision").innerHTML = 
          `<strong>Decision:</strong> ${data.decision}`
        
        document.getElementById("liveFrameCount").innerHTML = 
          `Frames analyzed: ${frameCount}`
        
        // Display captured frame
        document.getElementById("liveFrameCapture").src = 
          canvas.toDataURL('image/jpeg')
        
        // Display mask
        document.getElementById("liveMaskCapture").src = 
          "data:image/png;base64," + data.mask
        
        // Continue analysis loop (every 1.5 seconds)
        if (analysisRunning) {
          setTimeout(analysisLoop, 1500)
        }
        
      } catch (error) {
        console.error("Error in analysis loop:", error)
        if (analysisRunning) {
          setTimeout(analysisLoop, 1500)
        }
      }
    }, 'image/jpeg', 0.8)
    
  } catch (error) {
    console.error("Frame capture error:", error)
    if (analysisRunning) {
      setTimeout(analysisLoop, 1500)
    }
  }
}
