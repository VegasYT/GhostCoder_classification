import React, { useState } from "react"
import '../css/Banner.css'
import banner from '../img/development.svg'
import Preloader from "./Preloader"

const Banner:React.FC = () => {

    const [preloader, setPreloader] = useState<boolean>(false)

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        event.preventDefault();
        const files = Array.from(event.target.files || []);
    
        if (!files.length) {
            return;
        }

        setPreloader(true)
    
        const formData = new FormData();
        files.forEach(file => formData.append('files', file));
    
        fetch('http://127.0.0.1:8000/upload_file/', {
            method: 'POST',
            body: formData,
        })
        .then(response => {
            if (response.ok) {
                return response.blob();
            } else {
                console.error('Server response was not OK');
                throw new Error('Server response was not OK');
            }
        })
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'sorted.zip';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            setPreloader(false)
        })
        .catch(error => {
            console.error('Error uploading file:', error);
        });
    };
    

    return(
        <div className="warper">
            <Preloader active={preloader}/>
            <div className="banner">
                <div className="banner-box" id="bannerForText">
                    <div id="bannerBoxText">
                        <div className={`box-text`}>
                            <div className="div-banner-description">
                                <span className="banner-description">Нейросеть для работы с документами</span>
                            </div>
                            <div className="text-title">
                                <span className="banner-title-main"><span className="banner-title-main-bold">Классификация документов</span><br />Облегчи работу с документами!</span>
                            </div>
                            <div className="text-info">
                                <span className="info">При заполнении заявки дарим<br />бесплатный аудит вашего сайта!</span>
                            </div>
                        </div>
                        <div className="box-buttons">
                            <button className="box-but" id="start-but">Дообучить нейросеть</button>
                            <form>
                                <input type="file" onChange={handleFileChange} style={{display: 'none'}} id="file-input" multiple name='files' />
                                <label htmlFor="file-input" className="box-but" id="signup-but">Загрузить файлы</label>
                            </form>
                        </div>
                    </div>
                </div>
                <div className="banner-box" id="bannerForImage">
                    <img src={ banner } alt="" className="box-img" loading="lazy" />
                </div>
            </div>
        </div>
        
    )
}

export default Banner;