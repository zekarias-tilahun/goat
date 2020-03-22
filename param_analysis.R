library(tidyverse)

setwd('C:/workspace/projects/gap/results/')

mutate_frame <- function(frame, param){
  if (param == 'd') return(mutate(frame,  train_rate = as.factor(train_rate * 100), d = as.factor(d)))
  else return(mutate(frame,  train_rate = as.factor(train_rate * 100), Neighborhood = as.factor(Neighborhood)))
}

plot_lp <- function(c.names, param, y.label){
  pfx = 'param_analysis_'
  sfx = 'cora.txt'
  if (param == 'd') file.name = paste0(pfx, sfx)
  else file.name = paste0(pfx, 'nbr_size_', sfx)
  
  c_frame = read_delim(file.name, delim = ' ')
  colnames(c_frame) = c.names
  
  sfx = 'zhihu.txt'
  if (param == 'd') file.name = paste0(pfx, sfx)
  else file.name = paste0(pfx, 'nbr_size_', sfx)
  z_frame = read_delim(file.name, delim = ' ')
  colnames(z_frame) = c.names
  
  sfx = 'email.txt'
  if (param == 'd') file.name = paste0(pfx, sfx)
  else file.name = paste0(pfx, 'nbr_size_', sfx)
  e_frame = read_delim(file.name, delim = ' ')
  colnames(e_frame) = c.names
  e_frame = e_frame[(e_frame$data == 'email') & (e_frame$task == 'link_prediction'),]
  
  
  c_frame = mutate_frame(c_frame, param)
  z_frame = mutate_frame(z_frame, param)
  e_frame = mutate_frame(e_frame, param)
  
  df = bind_rows(c_frame, z_frame, e_frame)
  
  if (param == 'd') base = ggplot(df, aes(train_rate, AUC, fill=d))
  else base = ggplot(df, aes(train_rate, AUC, fill=Neighborhood))
  plt = base + 
    geom_bar(position = 'dodge', stat = 'identity') +
    geom_errorbar(aes(min=AUC - std, max=AUC + std), position = 'dodge') +
    labs(x='% of training edges', y=y.label) + 
    scale_fill_brewer(palette = "Dark2") +
    theme(panel.background = element_rect(fill = 'white'),
          panel.grid.major = element_line(color='black'),
          axis.line = element_line(color='black'),
          legend.title = element_text(size = 18),
          legend.text = element_text(size = 18),
          axis.text = element_text(size = 18),
          axis.title = element_text(size = 20),
          strip.background = element_rect(fill = "white", color = 'black'),
          strip.text = element_text(size = 24),
          legend.position = 'top'
    ) + facet_wrap(~data, scales = "free")
  return(plt)
}

plot_nc <- function(){
  e_dim_frame = read_delim('param_analysis_email.txt', delim = ' ')
  colnames(e_dim_frame) = c('NMI', 'AMI', 'd', 'train_rate', 'task', 'data')
  e_dim_frame = e_dim_frame[(e_dim_frame$data == 'email') & (e_dim_frame$task == 'node_clustering'),]
  e_dim_frame = mutate(e_dim_frame, train_rate = as.factor(train_rate * 100))
  e_dim_frame = e_dim_frame  %>% select(NMI:data) %>%  
    gather(key='Metric', value='measurement', NMI, AMI, -d, -train_rate) %>%
    gather(key = "param", value = 'value', d,  -train_rate, -measurement, -Metric)
  
  e_nbr_frame = read_delim('param_analysis_nbr_size_email.txt', delim = ' ')
  colnames(e_nbr_frame) = c('NMI', 'AMI', 'Neighborhood', 'train_rate', 'task', 'data')
  e_nbr_frame = e_nbr_frame[(e_nbr_frame$data == 'email') & (e_nbr_frame$task == 'node_clustering'),]
  e_nbr_frame = mutate(e_nbr_frame, train_rate = as.factor(train_rate * 100))
  e_nbr_frame = e_nbr_frame  %>% 
    select(NMI:data) %>%  gather(key='Metric', value='measurement', NMI, AMI, -Neighborhood, -train_rate) %>% 
    gather(key = "param", value = 'value', Neighborhood,  -train_rate, -measurement, -Metric)
  
  e_dim_frame
  e_nbr_frame
  df = bind_rows(e_dim_frame, e_nbr_frame)
  df = df[df$train_rate == 55,]
  df
  plt = ggplot(df, aes(x=value, y=measurement, color = Metric, group = Metric, shape=Metric)) + 
    geom_point(size=4) + geom_line(size=2) + 
    facet_wrap(~param, scales = "free", strip.position = "bottom") +
    theme(panel.background = element_rect(fill = 'white'),
          panel.grid.major = element_line(color='black'),
          axis.line = element_line(color='black'),
          legend.title = element_text(size = 18),
          legend.text = element_text(size = 18),
          axis.text = element_text(size = 18),
          axis.title = element_text(size = 20),
          strip.background = element_blank(),
          strip.text = element_text(size = 24),
          legend.position = 'top',
          strip.placement = "outside") +
    labs(x=NULL, y = 'Score')
  return(plt)
}

nc.plt = plot_nc()
ggsave('C:/workspace/projects/gap/gap-paper/img/param_analsis_nc.pdf', height = 3.5, width = 7, device = 'pdf', plot = nc.plt)

c.names = c('AUC', 'std', 'd', 'train_rate', 'task', 'data')
dim.plt = plot_lp(c.names = c.names, param = 'd', y.label = 'Garbage')
dim.plt
ggsave('C:/workspace/projects/gap/gap-paper/img/param_analsis_lp_dim.pdf', height = 4, width = 10, device = 'pdf', plot = dim.plt)


c.names = c('AUC', 'std', 'Neighborhood', 'train_rate', 'task', 'data')
nbr.plt = plot_lp(c.names = c.names, param = 'Neighborhood', y.label = 'AUC')
ggsave('C:/workspace/projects/gap/gap-paper/img/param_analsis_lp_nbr_size.pdf', height = 4, width = 10, device = 'pdf', plot = nbr.plt)


vis = read_delim('vis.txt', delim = ' ')
vis = mutate(vis, communities=as.factor(communities))
vis
vis.plt = ggplot(vis, aes(x=dim1, y=dim2, color=communities)) + geom_point(size=.5) + facet_wrap(~Algorithm, ncol = 5) + 
  theme(legend.position = "None", 
        panel.background = element_blank(), 
        axis.ticks = element_blank(), 
        axis.text = element_blank(),
        axis.title = element_blank(), 
        strip.background = element_rect(fill='white'),
        strip.text = element_text(size=20))
vis.plt
ggsave('C:/workspace/projects/gap/gap-paper/img/visualization.pdf', height = 3, width = 8, device = 'pdf', plot = vis.plt)


