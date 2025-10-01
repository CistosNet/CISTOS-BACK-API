document.addEventListener('DOMContentLoaded', function () {
    const analyzeBtn = document.getElementById('analyze-btn');
    const loadingSection = document.getElementById('loading-section');
    const resultsSection = document.getElementById('results-section');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const analysisName = document.getElementById('analysis-name');
    const analysisNameDisplay = document.getElementById('analysis-name-display');
    const newAnalysisBtn = document.getElementById('new-analysis-btn');
    const downloadBtn = document.getElementById('download-report-btn');
    const imageCounter = document.getElementById('image-counter');
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');

    let selectedFiles = [];
    let analysisResults = [];
    let reportUrl = null;
    let currentIndex = 0;

    analyzeBtn.addEventListener('click', startAnalysis);
    prevBtn.addEventListener('click', ()=>{ if(currentIndex>0){ currentIndex--; showResults(); } });
    nextBtn.addEventListener('click', ()=>{ if(currentIndex<analysisResults.length-1){ currentIndex++; showResults(); } });
    newAnalysisBtn.addEventListener('click', resetAnalysis);

    window.setSelectedFiles = function (files) { selectedFiles = files; }

    async function startAnalysis() {
        if(!analysisName.value.trim()){ alert('Por favor, dê um nome à sua análise.'); return; }
        if(selectedFiles.length===0){ alert('Nenhum arquivo foi selecionado.'); return; }

        document.getElementById('step-upload').classList.add('tw-hidden');
        document.getElementById('step-name').classList.add('tw-hidden');
        analyzeBtn.classList.add('tw-hidden');
        loadingSection.classList.remove('tw-hidden');
        updateProgress(20);

        try {
            const formData = new FormData();
            selectedFiles.forEach(f=>formData.append("files",f));
            formData.append("analysis_name", analysisName.value.trim());

            const response = await fetch("/predict/policistos", { method:"POST", body: formData });
            if(!response.ok) throw new Error("Erro no processamento: "+response.status);

            const data = await response.json();
            analysisResults = data.results || [];
            reportUrl = data.report_url || null;
            currentIndex = 0;
            showResults();
        } catch(err){
            console.error(err); alert('Erro ao processar a análise.');
        }
    }

    function updateProgress(percent){
        progressBar.style.width = percent+'%';
        progressText.textContent = percent+'% completo';
    }

    function showResults(){
        loadingSection.classList.add('tw-hidden');
        resultsSection.classList.remove('tw-hidden');
        analysisNameDisplay.textContent = analysisName.value.trim();
        if(analysisResults.length===0) return;

        const current = analysisResults[currentIndex];
        document.getElementById('total-follicles').textContent = current.info?.cistos ?? "--";

        const resultsContainer = document.getElementById('result-images');
        resultsContainer.innerHTML='';
        const img = document.createElement('img');
        img.src = current.image;
        img.alt = "Resultado da análise";
        img.className = "tw-w-full tw-h-auto tw-rounded-lg tw-shadow";
        resultsContainer.appendChild(img);

        imageCounter.textContent = `Imagem ${currentIndex+1} de ${analysisResults.length}`;

        prevBtn.disabled = currentIndex===0;
        nextBtn.disabled = currentIndex===analysisResults.length-1;

        if(reportUrl){ downloadBtn.href=reportUrl; downloadBtn.download=`Resultados_${analysisName.value.trim()}.zip`; downloadBtn.style.display='inline-block'; }
        else downloadBtn.style.display='none';
    }

    function resetAnalysis(){
        selectedFiles = [];
        analysisResults = [];
        currentIndex = 0;
        reportUrl = null;

        document.getElementById('step-upload').classList.remove('tw-hidden');
        document.getElementById('step-name').classList.add('tw-hidden');
        analyzeBtn.classList.add('tw-hidden');
        loadingSection.classList.add('tw-hidden');
        resultsSection.classList.add('tw-hidden');
    }
});