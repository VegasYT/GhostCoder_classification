import React, {useRef} from "react"
import Slider from "react-slick"
import '../css/Advantages.css'
import adv2 from '../img/advantage1.png'
import adv3 from '../img/advantage9.png'
import adv4 from '../img/advantage10.png'
import adv6 from '../img/advantage11.png'
import adv7 from '../img/advantage2.png'
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";


const advantages = [
    {
        title: 'Современные технологии',
        description: 'Мы используем современные технологии, среди которых PyTorch, React и Django',
        img: adv2
    },
    {
        title: 'Работа с несколькими файлами',
        description: 'Вам не нужно отправлять постоянно по 1 документу, можете сразу загрузить несколько',
        img: adv4
    },
    {
        title: 'Сортировка ваших документов',
        description: 'После классификации документа, вам не нужно их потом сортировать, мы сделали это за вас',
        img: adv3
    },
    {
        title: 'Простота в использовании',
        description: 'Вам достаточно лишь загрузить файлы и дождаться скачивания отсортированного архива',
        img: adv7
    },
    {
        title: 'Возможность дообучения',
        description: 'Если ваш тип документа не поддерживается, вы можете дообучить нейросеть',
        img: adv6
    },
]

const Advantages:React.FC = () => {

    const slideRef = useRef<Slider | null>(null)

    const sliderSettings = {
        dots: false,
        infinite: true,
        speed: 500,
        slidesToShow: 3,
        slidesToScroll: 1,
        swipeToSlide: true,
        arrows: false,
        centerMode: false,
        responsive: [
            {
                breakpoint: 1068,
                settings: {
                    slidesToShow: 2
                }
            },
            {
                breakpoint: 721,
                settings: {
                    slidesToShow: 1,
                }
            }
        ]
    }

    const on_next = () => {
        slideRef.current?.slickNext()
    }

    const on_prev = () => {
        slideRef.current?.slickPrev()
    }

    return (
        <div className="advantages-container">
            <div className="div-advantages-title-container">
                <div className="div-advantages-title">
                    <span className="advantages-title">Почему мы?</span>
                </div>
                <div className="advantages-div-buts">
                    <div className="advantages-div-but">
                        <span className="advantages-but" onClick={on_prev}>←</span>
                    </div>
                    <div className="advantages-div-but">
                        <span className="advantages-but" onClick={on_next}>→</span>
                    </div>
                </div>
            </div>
            <div className="div-advantages-blocks">
                <Slider {...sliderSettings} ref={slideRef}>
                    {advantages.map((item, index) => (
                        <div className="div-advantage-block" key={index}>
                            <div className="div-advantage-block-img">
                                <img src={item.img} alt="" className="advantage-block-img" />
                            </div>
                            <div className="div-advantage-block-title">
                                <span className="advantage-block-title">{item.title}</span>
                            </div>
                            <div className="div-advantage-block-description">
                                <span className="advantage-block-description">{item.description}</span>
                            </div>
                        </div>
                    ))}
                </Slider>
            </div>
        </div>
    )
}

export default Advantages