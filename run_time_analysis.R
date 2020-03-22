library(tidyverse)

setwd('C:/workspace/projects/gap/results/')

df = read_delim('./run_time.txt', delim = " ")

plt = ggplot(df, aes(x=Dataset, y=Time, fill=Algorithm)) + 
  geom_bar(position = 'dodge', stat = "identity") + 
  geom_text(aes(label=Time), vjust=0, stat = 'identity', position = position_dodge(width = 1)) + 
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
  labs(y='Time (in seconds)')
  scale_fill_brewer(palette = "Dark2")

ggsave(filename = 'C:/workspace/projects/gap/gap-paper/img/run_time.pdf', device = "pdf", height = 4, width = 5)
