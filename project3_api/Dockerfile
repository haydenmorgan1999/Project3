FROM rstudio/plumber 

WORKDIR /app

RUN apt-get update -qq && apt-get install -y libssl-dev libcurl4-gnutls-dev libpng-dev pandoc

RUN R -e "install.packages(c(plumber', 'dplyr', 'tidyverse', 'tidymodels', 'caret', 'yardstick', 'ranger'))"

COPY plumber.R plumber.R

COPY data/ /app/data/

EXPOSE 8000

ENTRYPOINT ["R", "-e", \ 
"pr <- plumber::plumb('plumber.R'); pr$run(host='0.0.0.0', port=8000)"]