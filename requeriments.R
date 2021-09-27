install.packages("devtools", repos='http://cran.rstudio.com/')
install.packages("plotly", repos='http://cran.rstudio.com/') 

install.packages("ellipsis", version = "0.3.2", repos='http://cran.rstudio.com/') 



library('devtools')
install_github("RBigData/pbdZMQ")

install.packages("dplyr",  repos = 'https://cloud.r-project.org')
install.packages("fNonlinear", repos='http://cran.rstudio.com/')
install.packages("ppclust",  repos='http://cran.rstudio.com/')
install.packages("EMD", repos='http://cran.rstudio.com/')
install.packages("tseriesChaos", repos='http://cran.rstudio.com/')
install.packages("Metrics", repos='http://cran.rstudio.com/')
install.packages("beepr", repos='http://cran.rstudio.com/')

install.packages("tidyverse", repo = 'https://mac.R-project.org')
install.packages("rgl", repos='http://R-Forge.R-project.org') # no instaled
install.packages("TSdist",  repos='http://cran.rstudio.com/') # no instaled

# for jupyter-lab
devtools::install_github('IRkernel/IRkernel')
devtools::install_github("IRkernel/repr")
devtools::install_github("IRkernel/IRdisplay", force = TRUE)
devtools::install_github("IRkernel/IRkernel")

# install system-wide
IRkernel::installspec(user = FALSE)