# -------------  LOAD LIBRARIES -------------
instalar <- function(paquete) {
  if (!require(paquete,character.only = TRUE, quietly = TRUE, warn.conflicts = FALSE)) {
    install.packages(as.character(paquete), dependecies = TRUE, repos = "http://cran.us.r-project.org")
    library(paquete, character.only = TRUE, quietly = TRUE, warn.conflicts = FALSE)
  }
}

paquetes <- c('scales', "DBI", "RPostgres", "reshape2", "DT",
              "rstudioapi", "lubridate", "RColorBrewer", "plyr",
              "zoo", "dbplyr","tidyr","knitr",'ggplot2','readr',
              "tidyverse",'igraph',  "ggraph", "tidygraph",'ggpubr', 'ggalluvial', 'glue')

lapply(paquetes, instalar)

# -------------  DB CONNECTION -------------

credentials <- read_csv("credentials.csv")

# conneción con dplyr
con <- dbConnect(RPostgres::Postgres(),
                 dbname = credentials$dbname,
                 host = credentials$host,
                 port = credentials$port,
                 user = credentials$user,
                 password = credentials$password
)

query_db <- function(query){
  rs <- dbSendQuery(con, statement=query)
  dbFetch(rs, n= -1)
}

get_data <-
  function(query){
    on.exit(dbDisconnect(conn)) ## important to close connection
    conn <- dbConnect(RPostgres::Postgres(),
                      dbname = credentials$dbname,
                      host = credentials$host,
                      port = credentials$port,
                      user = credentials$user,
                      password = credentials$password)
    dbGetQuery(con,query)
  }


# -------------  Graphs  -------------
#source("./../helpers/timelines.r")
