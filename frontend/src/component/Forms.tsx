import React, { useState, useEffect } from "react"
import { useToast } from "@chakra-ui/react"
import '../css/Forms.css'

const Forms:React.FC = () => {
    
    const toast = useToast()

    const [classes, setClasses] = useState<{ [key: string]: number }>({});

    useEffect(() => {
        fetchClasses();
    }, []);

    const fetchClasses = async () => {
        try {
            const response = await fetch('http://127.0.0.1:8000/get_classes/');
            if (!response.ok) {
                throw new Error('Failed to fetch classes');
            }
            const data = await response.json();
            setClasses(data);
        } catch (error) {
            console.error('Error fetching classes:', error);
        }
    };

    const saveClassCounts = async () => {
        try {
            const response = await fetch('http://127.0.0.1:8000/save_class_counts/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(classes)
            });
            if (!response.ok) {
                throw new Error('Failed to save class counts');
            }
            toast({
                title: 'Данные сохранены',
                status: 'success',
                duration: 3000,
                isClosable: true,
            })
        } catch (error) {
            console.error('Error saving class counts:', error);
        }
    };

    const handleChange = (key: string, value: number) => {
        setClasses((prevClasses) => ({
          ...prevClasses,
          [key]: value,
        }));
      };

    return (
        <div className="forms-container">
            <div className="div-forms-main">
                <div className="div-forms">
                    <div className="div-form">
                        <div className={`div-form-title`}>
                            {Object.entries(classes).map(([key, _]) => (
                                <span className="form-title" key={key}>{key}</span>
                            ))}
                        </div>
                    </div>
                    <div className="div-form">
                        <div className="div-forms-input">
                            <div className="div-forms-input-container">
                                {Object.entries(classes).map(([key, value]) => (
                                    <div className="div-form-input">
                                        <input type="number" className="form-input" key={key} defaultValue={value} onChange={(e) => handleChange(key, parseInt(e.target.value))}/>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
                <div className="div-form-but">
                    <button className="form-but" onClick={saveClassCounts}>Сохранить</button>
                </div>
            </div>
        </div>
    )
}

export default Forms