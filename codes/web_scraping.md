Web Sraping
================

## Scraping James (LOL)

``` r
library(tidyverse)
library(rvest)
library(knitr)
```

``` r
url<-read_html('https://es.wikipedia.org/wiki/Anexo:Partidos_de_la_selecci%C3%B3n_de_f%C3%BAtbol_de_Colombia#A%C3%B1os_2020_-_2029')

marcadores<-url%>%
  html_nodes('td:nth-child(6) > b')%>%
  html_text()%>%
  tbl_df()%>%
  filter(str_detect(value,':'))%>%
  rename('Marcadores'=value)

Fecha<-url%>%
  html_nodes('td:nth-child(3)')%>%
  html_text()%>%
  tbl_df()%>%
  filter(str_detect(value,'de'))%>%
  mutate(value=str_replace_all(value,"\n|\r", ""))%>%
  mutate(value=str_replace_all(value, "[^[:alnum:]]", " "))%>%
  rename('Fecha'=value)
  
rival<-url%>%
  html_nodes('td:nth-child(5) > a')%>%
  html_text()%>%
  tbl_df()%>%
  rename("Rival"=value)

base_colombia<-data.frame(Fecha,rival,marcadores)
base_partidos<-base_colombia%>%
  filter(str_detect(Fecha,'2019|2020'))%>%
  separate(Marcadores,c('Marcador_Colombia','Marcador_Rival'),sep=':',fill='right')%>%
  mutate(Marcador_Colombia = as.numeric(Marcador_Colombia),
         Marcador_Rival = as.numeric(Marcador_Rival),
         Resultado = case_when(Marcador_Colombia>Marcador_Rival~"Gano",
                               Marcador_Colombia<Marcador_Rival~"Perdió",
                               Marcador_Colombia==Marcador_Rival~"Empato"))


base_partidos%>%
  kable(format = 'markdown', caption = 'Scraping Partidos')
```

| Fecha                    | Rival         | Marcador\_Colombia | Marcador\_Rival | Resultado |
| :----------------------- | :------------ | -----------------: | --------------: | :-------- |
| 22 de marzo de 2019      | Japón         |                  1 |               0 | Gano      |
| 26 de marzo de 2019      | Corea del Sur |                  1 |               2 | Perdió    |
| 3 de junio de 2019       | Panamá        |                  3 |               0 | Gano      |
| 9 de junio de 2019       | Perú          |                  3 |               0 | Gano      |
| 15 de junio de 2019      | Argentina     |                  2 |               0 | Gano      |
| 19 de junio de 2019      | Catar         |                  1 |               0 | Gano      |
| 23 de junio de 2019      | Paraguay      |                  1 |               0 | Gano      |
| 28 de junio de 2019      | Chile         |                  0 |               0 | Empato    |
| 6 de septiembre de 2019  | Brasil        |                  2 |               2 | Empato    |
| 10 de septiembre de 2019 | Venezuela     |                  0 |               0 | Empato    |
| 10 de octubre de 2019    | Chile         |                  0 |               0 | Empato    |
| 15 de octubre de 2019    | Argelia       |                  0 |               3 | Perdió    |
| 15 de noviembre de 2019  | Perú          |                  1 |               0 | Gano      |
| 19 de noviembre de 2019  | Ecuador       |                  1 |               0 | Gano      |
| 9 de octubre de 2020     | Venezuela     |                  3 |               0 | Gano      |
| 13 de octubre de 2020    | Chile         |                  2 |               2 | Empato    |
| 13 de noviembre de 2020  | Uruguay       |                  0 |               3 | Perdió    |
| 17 de noviembre de 2020  | Ecuador       |                  1 |               6 | Perdió    |

Scraping Partidos
