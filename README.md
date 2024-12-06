Для работы требуется установка библиотеки [torchdiffeq](https://github.com/rtqichen/torchdiffeq):
```
pip install torchdiffeq
```

Основа кода взята из того же репозитория, файла с демонстрацией работы [ode_demo.py](https://github.com/rtqichen/torchdiffeq/blob/master/examples/ode_demo.py).
Запуск эксперимента производится командой

```
python code/main.py --viz --data_type=spiral  --max_itr=500
```
В аргументе `data_type` указывается тип особой точки, поддерживаемые: `uniform`, `saddle`, `center`, `spiral`; Аргумент `max_itr` задает количество итераций обучения.

Gif с результатом работы были получены командой
```
convert -delay 10 -loop 0 code/spiral-gif/frame-* code/spiral.gif
```

В $\LaTeX$ презентации gif были получены с помощью библиотеки [animate](https://ctan.org/pkg/animate?lang=en).
