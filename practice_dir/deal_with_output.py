import matplotlib.pyplot as plt
import glob, os
import json

def get_percentile(list_of_lists, what_percent):
#         assume all lists in the list of lists are of the same length
        overall_perc = []
        length = len(list_of_lists[0])
        print('length of list is ' + str(length))
        for position in range(length):
            this_pos = []
            for this_list in list_of_lists:
                if len(this_list) != length:
                    print('We have a list with length ' + str(len(this_list)))
                if len(this_list) == length and this_list[position] != None:
                   this_pos.append(this_list[position])
            this_pos = sorted(this_pos)
            perc_pos = int(round(what_percent*(len(this_pos)-1), 0))
            # print(perc_pos)
            overall_perc.append(this_pos[perc_pos])
        return overall_perc
    
    
def get_repro_by_start_string(start_string):
    practice_dir = '.'
    
    list_of_outcomes = []
    for file in os.listdir(practice_dir):
      if file.startswith(start_string) and file.endswith('.json'):
        with open(file) as fin :
           structure = json.loads(fin.read())
           list_of_outcomes.extend(structure)
    return list_of_outcomes
    

def get_state_by_start_string(start_string, key_of_interest = 'cum_cases'):
    practice_dir = '.'
    list_of_outcomes = []
    for file in os.listdir(practice_dir):
      if file.startswith(start_string) and file.endswith('.json'):
        with open(file) as fin :
           list_of_state_dicts = json.loads(fin.read())
           for state_dict in list_of_state_dicts:
            for key in state_dict:
                if key == key_of_interest:
                    list_of_outcomes.append(state_dict[key])
    return list_of_outcomes
    

def plotting_repro():
    plt.clf()
    save_string = 'sample_repro.pdf'
    weekly_string = 'reproductive_trial-result_household_schedule_weekly'
    biweekly_string = 'reproductive_trial-result_household_schedule_biweekly'
    no_test_string = 'reproductive_trial-result_No_test_never'
    label_string  = {weekly_string:'Weekly household testing', biweekly_string:'Weekly half-household testing', no_test_string: 'No asymptomatic testing'}
    colour_string  = {weekly_string:'blue', biweekly_string:'orange', no_test_string: 'red'}
    
    weeks_to_cutoff = 2
    
    strings = [weekly_string, biweekly_string, no_test_string]
    for string in strings:
        this_repro = get_repro_by_start_string(string)
        upper = get_percentile(this_repro, 0.975)
        lower = get_percentile(this_repro, 0.025)
        med = get_percentile(this_repro, 0.5)
        
        upper = upper[:len(upper)-weeks_to_cutoff]
        lower = lower[:len(lower)-weeks_to_cutoff]
        med = med[:len(med)-weeks_to_cutoff]
        
        
        test_style_colour = colour_string[string]
        test_style_string = label_string[string]
        
        plt.plot(range(len(med)), med, label=test_style_string, color = test_style_colour)
        plt.fill_between(range(len(lower)), upper, y2 = lower, color = test_style_colour, alpha = 0.2)
    plt.legend()
    plt.xlabel('Week')
    plt.ylabel('Estimated effective reproductive number')
    plt.savefig(save_string)
    
def plotting_cum_cases():
    plt.clf()
    save_string = 'sample_cum_cases.pdf'
    weekly_string = 'state_trial-result_household_schedule_weekly'
    biweekly_string = 'state_trial-result_household_schedule_biweekly'
    no_test_string = 'state_trial-result_No_test_never'
    label_string  = {weekly_string:'Weekly household testing', biweekly_string:'Weekly half-household testing', no_test_string: 'No asymptomatic testing'}
    colour_string  = {weekly_string:'blue', biweekly_string:'orange', no_test_string: 'red'}
    
    
    strings = [weekly_string, biweekly_string, no_test_string]
    for string in strings:
        this_repro = get_state_by_start_string(string)
        upper = get_percentile(this_repro, 0.975)
        lower = get_percentile(this_repro, 0.025)
        med = get_percentile(this_repro, 0.5)
        
        test_style_colour = colour_string[string]
        test_style_string = label_string[string]
        
        plt.plot(range(len(med)), med, label=test_style_string, color = test_style_colour)
        plt.fill_between(range(len(lower)), upper, y2 = lower, color = test_style_colour, alpha = 0.2)
    plt.legend()
    plt.xlabel('Day')
    plt.ylabel('Cumulative cases')
    plt.savefig(save_string)
    
# result = get_state_by_start_string('state', key_of_interest = 'cum_cases')
# for item in result:
#     print(item)
    
plotting_repro()
plotting_cum_cases()

            # if test_style == 'household_schedule':
            #     if denom == 1:
            #         test_style_string = 'Weekly household testing'
            #         test_style_colour = 'blue'
            #     if denom == 2:
            #         test_style_string = 'Weekly half-household testing'
            #         test_style_colour = 'orange'
            # elif test_style == 'No_test':
            #     test_style_string = 'No asymptomatic testing'
            #     test_style_colour = 'red'
            # elif test_style == None:
            #     test_style_string = 'Random asymptomatic testing every two weeks'
            #     test_style_colour = 'grey'
            # 
            # for line  in lines_of_interest:
            #         axes.set_ylim([0, max_y])
            #         axes.plot(range(len(results[line])), results[line], label=test_style_string, color = test_style_colour)
            #         axes.fill_between(range(len(top[line])), top[line], y2 = bottom[line], color = test_style_colour, alpha = 0.2)
            #         
            #         print(line + str(results[line]))
            #         print(line + str(top[line]))
            # # axes.plot(true_x, true_y, color = 'black', label='Data')
            # axes.legend()
            # axes.set_ylim([0, max_y])
            # axes.set(xlabel = 'Day')
            # axes.set(ylabel = 'Number of cumulative cases')
            # axes.set_title('Cumulative cases over time under differing testing strategies')
            # # plt.savefig('model_output_graph-' +graph_type + "_testingProb-" + str(testProb) + "tracing_eff-" + str(tracing_efficiency) + '.png')
            # 
            # repro_name = 'repro_rate'
            # repro_med = results[repro_name]
            # ax2.plot(range(len(repro_med)), repro_med, label=test_style_string, color = test_style_colour)
            # ax2.fill_between(range(len(top[repro_name])), top[repro_name], y2 = bottom[repro_name], color = test_style_colour, alpha = 0.2)



#     
# prac_file = 'reproductive_trial-result_No_test_never_2021-02-18T14:32:19.690836_.json'
# with open(prac_file) as fin :
#     print(json.loads(fin.read()))
    
# start = 'reproductive'
# get_by_start_string(start)