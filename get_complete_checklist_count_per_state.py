


import matplotlib.pyplot as plt
%matplotlib inline



all_state_count = df_usa_count[["STATE", "n"]]


state_count = all_state_count.groupby(["STATE"]).n.sum().reset_index()
state_count = state_count.sort_values(by=['n'], ascending = False)
len(state_count)

plot_title = 'Number of checklist (i.e. n) in each state of US'
fig_title  =  'state_wise_checklist_count.png'

plt.figure()
plt.rcParams.update() # must set in top

ax = state_count.plot(kind = 'barh',x = 'STATE', y= 'n', 
                  figsize=(10,100), grid = True)

ax.set_xlabel('Number of checklist', fontdict={'fontsize':18})
ax.set_ylabel('US State'  , fontdict={'fontsize':18})
ax.legend(bbox_to_anchor=(1.35, 1), loc = 'upper right', fontsize=20)
ax.set_title(plot_title, fontsize=15)

# create a list to collect the plt.patches data
totals = []

# find the values and append to list
for i in ax.patches:
    totals.append(i.get_width())

# set individual bar lables using above list
total = sum(totals)

# set individual bar lables using above list
for i in ax.patches:
   # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width()+.1, i.get_y()+.31, \
            str(round((i.get_width()), 2)), 
            fontsize=10, color='red')

# invert for largest on top 
ax.invert_yaxis()

print("Number of states in the plot:", len(state_count))
plt.savefig(fig_title, format='png', bbox_inches = 'tight', facecolor= 'white', transparent=True)