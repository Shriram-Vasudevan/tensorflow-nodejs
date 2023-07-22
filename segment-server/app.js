const express = require('express');
const multer = require('multer');
const { exec } = require('child_process');
const fs = require('fs');

const app = express();
const upload = multer(); 

app.post('/segment-image', upload.single('file'), (req, res) => {
    let img = req.file.buffer;

    fs.writeFile('temp.img', img, (error) => {
        if (error) {
            console.error(`writeFile error: ${error}`);
            return res.status(500).send('Error writing file');
        }

        exec('python3 ObjectClassification.py', (error, stdout, stderr) => {
            if (error) {
                console.error(`exec error: ${error}`);
                return res.status(500).send('Error executing Python script');
            }
            console.error(`stderr: ${stderr}`);
            res.status(200).send(stdout);
        });
    });
});

app.listen(3000, () => console.log('Server running on port 3000'));
