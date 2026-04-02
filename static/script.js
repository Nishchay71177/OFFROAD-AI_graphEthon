let file

const dropZone=document.getElementById("dropZone")

const input=document.getElementById("fileInput")

dropZone.onclick=()=>input.click()

input.onchange=e=>{

file=e.target.files[0]

previewImage(file)

}

dropZone.ondragover=e=>{

e.preventDefault()

dropZone.style.background="rgba(56,189,248,0.15)"

}

dropZone.ondrop=e=>{

e.preventDefault()

file=e.dataTransfer.files[0]

previewImage(file)

}

function previewImage(file){

let reader=new FileReader()

reader.onload=e=>{

document.getElementById("preview").src=e.target.result

}

reader.readAsDataURL(file)

}



async function predict(){

let formData=new FormData()

formData.append("file",file)

document.getElementById("decision").innerHTML="Analyzing..."

let res=await fetch("/predict-image",{

method:"POST",

body:formData

})

let data=await res.json()

document.getElementById("terrain").innerHTML=

"Terrain: "+data.terrain

document.getElementById("confidence").innerHTML=

"Confidence: "+data.confidence+"%"

document.getElementById("decision").innerHTML=

"Navigation: "+data.decision


document.getElementById("maskPreview").src=

"data:image/png;base64,"+data.mask

}