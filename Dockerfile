FROM python:3.6

RUN echo export PATH=$HOME/.local/bin:$PATH >> ~/.bashrc

COPY requirements.txt requirements.txt


RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y make
RUN apt-get install -y texlive-latex-base
RUN apt-get install -y texlive-fonts-recommended
#RUN apt-get install -y texlive-fonts-extra
RUN git clone --recursive https://github.com/andreasvc/disco-dop.git
RUN cd disco-dop


RUN pip install --user --no-cache-dir -r requirements.txt 

WORKDIR /disco-dop/
RUN pip install sphinx
RUN mkdir -p _static
RUN make install
RUN pip install grapheme

WORKDIR /home/merug/activedopmeru/
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=development

COPY . .

EXPOSE 5000
RUN python -m flask initdb
RUN python -m flask initpriorities

CMD ["python", "-m", "flask", "run", "--with-threads"]
