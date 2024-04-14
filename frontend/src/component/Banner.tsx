import React, { useState } from "react"
import { useToast, Box, useDisclosure, ModalOverlay, Modal, ModalContent, ModalHeader, ModalCloseButton, ModalBody, ModalFooter, Button } from '@chakra-ui/react'
import '../css/Banner.css'
import banner from '../img/development.svg'
import Preloader from "./Preloader"
import image1 from '../img/image1.jpg'
import image2 from '../img/image2.jpg'

const Banner:React.FC = () => {

    const OverlayOne = () => (
        <ModalOverlay
          bg='blackAlpha.300'
          backdropFilter='blur(10px)'
        />
      )

    const { isOpen, onOpen, onClose } = useDisclosure()

    const [hierarchy, setHierarchy] = useState<boolean>(false)

    const [preloader, setPreloader] = useState<boolean>(false)

    const toast = useToast()

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        event.preventDefault();
        const files = Array.from(event.target.files || []);
    
        if (!files.length) {
            return;
        }

        setPreloader(true)
    
        const formData = new FormData();
        files.forEach(file => formData.append('files', file));
        formData.append('hierarchy', String(hierarchy))
    
        fetch('http://127.0.0.1:8000/upload_file/', {
            method: 'POST',
            body: formData,
        })
        .then(response => {
            if (response.ok) {
                // Если ответ успешен, проверяем, есть ли ошибки в заголовках
                event.target.value = '';
                const errorMessage = response.headers.get('X-Error-Message');
                if (errorMessage){
                    const errorData = JSON.parse(errorMessage);
                    if (errorData.errors.length !== 0) {
                        const errorMessageText = errorData.errors.join('\n');
                        toast({
                            title: 'Ошибка',
                            description: (
                                <Box whiteSpace={"pre-line"}>
                                    {errorMessageText}
                                </Box>
                            ),
                            status: 'error',
                            isClosable: true,
                            duration: 9000,
                            position: 'bottom',
                        })
                    } else {
                        toast({
                            title: 'Успешно',
                            status: 'success',
                            isClosable: true,
                            duration: 3000,
                            position: 'bottom',
                        })
                    }
                }
                
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

    const handleRetrain = (event: React.ChangeEvent<HTMLInputElement>) => {
        event.preventDefault();
        const file = event.target.files && event.target.files[0];
    
        if (!file) {
            return;
        }

        setPreloader(true);
    
        const formData = new FormData();
        formData.append('csv_file', file);
    
        fetch('http://127.0.0.1:8000/upload_csv/', {
            method: 'POST',
            body: formData,
        })
        .then(response => {
            if (response.ok) {
                // Handle successful response
            } else {
                console.error('Server response was not OK');
                throw new Error('Server response was not OK');
            }
        })
        .catch(error => {
            console.error('Error retraining network:', error);
        })
        .finally(() => {
            setPreloader(false);
        });
    };

    return(
        <div className="warper">
            <Modal isCentered isOpen={isOpen} onClose={onClose}>
            <Preloader active={preloader}/>
                <OverlayOne />
                <ModalContent>
                    <ModalHeader>Выберите иерархию</ModalHeader>
                    <ModalCloseButton />
                    <ModalBody>
                        <div style={{display: 'flex', alignItems: 'center', flexDirection: 'column'}}>
                            <button onClick={() => {setHierarchy(false)}}>
                                <img src={image1} alt="" className={`modal-img ${hierarchy ? '' : 'select'}`}/>
                            </button>
                            <button onClick={() => {setHierarchy(true)}}>
                                <img src={image2} alt="" className={`modal-img ${hierarchy? 'select': ''}`}/>
                            </button>
                        </div>
                    </ModalBody>
                    <ModalFooter>
                        <Button onClick={onClose}>Close</Button>
                    </ModalFooter>
                </ModalContent>
            </Modal>
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
                            <form>
                                <input type="file" onChange={handleRetrain} style={{display: 'none'}} id="csv-input" name='csv_file' />
                                <label htmlFor="csv-input" className="box-but" id="start-but">Дообучить нейросеть</label>
                            </form>
                            <form>
                                <input type="file" onChange={handleFileChange} style={{display: 'none'}} id="file-input" multiple name='files' />
                                <label htmlFor="file-input" className="box-but" id="signup-but">Загрузить файлы</label>
                            </form>
                        </div>
                        <label className="box-but" id="start-but-2" onClick={() => {onOpen()}}>Выбрать иерархию</label>
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