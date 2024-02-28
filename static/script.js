document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('uploadForm').addEventListener('submit', function (event) {
        event.preventDefault(); // Prevent the default form submission

        var formData = new FormData();
        var fileInput = document.getElementById('fileInput');

        if (fileInput.files.length === 0) {
            alert('Please select a file.');
            return;
        }

        formData.append('file', fileInput.files[0]);

        fetch('/', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            displayImage(data);
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
});

function displayImage(data) {
    document.getElementById('inputImage').src = 'data:image/jpeg;base64,' + data.input_image;
    document.getElementById('concatenatedImage').src = 'data:image/jpeg;base64,' + data.concatenated_image;
    document.getElementById('binArea').innerText = 'Bin Area (mm^2): ' + data.bin_mm;
    document.getElementById('nbdArea').innerText = 'NBD Area (mm^2): ' + data.nbd_mm;
    document.getElementById('contaminationPercentage').innerText = 'Contamination Percentage: ' + data.contamination_percent + '%';

    document.getElementById('imageContainer').style.display = 'block'; // Show the image container
}
