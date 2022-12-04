FROM public.ecr.aws/lambda/python:3.9

COPY ["requirements.txt", "./"]

RUN pip install -r ./requirements.txt
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/blob/main/tflite/tflite_runtime-2.7.0-cp39-cp39-linux_x86_64.whl?raw=true

COPY ["lambda_function.py", "model.tflite", "./"]

CMD [ "lambda_function.lambda_handler"]
