document.addEventListener('DOMContentLoaded', function() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const folderInput = document.getElementById('folder-input');
    const selectFilesBtn = document.getElementById('select-files-btn');
    const selectFolderBtn = document.getElementById('select-folder-btn');
    const fileListContainer = document.getElementById('file-list-container');
    const fileList = document.getElementById('file-list');
    const stepName = document.getElementById('step-name');
    const analyzeBtn = document.getElementById('analyze-btn');

    let selectedFiles = [];

    ['dragenter','dragover','dragleave','drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e){ e.preventDefault(); e.stopPropagation(); }

    ['dragenter','dragover'].forEach(e => dropzone.addEventListener(e, ()=>dropzone.classList.add('tw-active')));
    ['dragleave','drop'].forEach(e => dropzone.addEventListener(e, ()=>dropzone.classList.remove('tw-active')));

    dropzone.addEventListener('drop', e => handleFiles(e.dataTransfer.files));
    selectFilesBtn.addEventListener('click', () => fileInput.click());
    selectFolderBtn.addEventListener('click', () => folderInput.click());
    fileInput.addEventListener('change', () => { if(fileInput.files.length) handleFiles(fileInput.files); });
    folderInput.addEventListener('change', () => { if(folderInput.files.length) handleFiles(folderInput.files); });

    function handleFiles(files){
        selectedFiles = Array.from(files).filter(file => {
            const name = file.name.toLowerCase();
            return (
                file.type.match('image.*') || 
                file.type.match('video.*') || 
                name.endsWith('.dcm') ||
                name.endsWith('.jpg') || 
                name.endsWith('.jpeg') || 
                name.endsWith('.png') ||
                name.endsWith('.mp4') || 
                name.endsWith('.avi') || 
                name.endsWith('.mov') || 
                name.endsWith('.mkv')
            );
        });

        if(selectedFiles.length>0){
            updateFileList();
            fileListContainer.classList.remove('tw-hidden');
            stepName.classList.remove('tw-hidden');
            analyzeBtn.classList.remove('tw-hidden');
        }
        if(window.setSelectedFiles) window.setSelectedFiles(selectedFiles);
    }

    function updateFileList(){
        fileList.innerHTML='';
        selectedFiles.forEach((file,index)=>{
            const fileItem = document.createElement('div');
            fileItem.className='file-item tw-p-3 tw-flex tw-items-center';
            const icon = document.createElement('i');
            icon.className='fas fa-file-image tw-text-indigo-500 tw-mr-3 tw-text-lg';
            icon.style.color='#64d4a4';
            const fileInfo = document.createElement('div'); fileInfo.className='tw-flex-1';
            const fileName = document.createElement('p'); fileName.className='tw-text-sm tw-font-medium tw-text-gray-800 tw-truncate'; fileName.textContent=file.name;
            const fileSize = document.createElement('p'); fileSize.className='tw-text-xs tw-text-gray-500'; fileSize.textContent=formatFileSize(file.size);
            fileInfo.appendChild(fileName); fileInfo.appendChild(fileSize);
            const removeBtn = document.createElement('button'); removeBtn.className='tw-text-red-500 hover:tw-text-red-700';
            removeBtn.innerHTML='<i class="fas fa-times"></i>';
            removeBtn.addEventListener('click', ()=>removeFile(index));
            fileItem.appendChild(icon); fileItem.appendChild(fileInfo); fileItem.appendChild(removeBtn);
            fileList.appendChild(fileItem);
        });
    }

    function removeFile(index){
        selectedFiles.splice(index,1);
        if(selectedFiles.length===0){
            fileListContainer.classList.add('tw-hidden');
            stepName.classList.add('tw-hidden');
            analyzeBtn.classList.add('tw-hidden');
        } else updateFileList();
    }

    function formatFileSize(bytes){
        if(bytes===0) return '0 Bytes';
        const k=1024; const sizes=['Bytes','KB','MB','GB'];
        const i=Math.floor(Math.log(bytes)/Math.log(k));
        return parseFloat((bytes/Math.pow(k,i)).toFixed(2))+' '+sizes[i];
    }

    window.getSelectedFiles=()=>selectedFiles;
});