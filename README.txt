**********************************************************************************
**********************  PART 1 - InteractiveExploration.py  **********************



Upload instance with the same format as "voitures.csv" :

    m = Modele(filename_instance)


----------------------------------------------------------------------------------

Get nearest solution to ideal point :
    - epsilon : (default = 0.1)
    m.nearest_point_id()


----------------------------------------------------------------------------------

Get nearest solution to a given reference point :
    - weight_file : parameter of the augmented weighted Tchebycheff method (default : weight_file.csv)
    - performance_file : data's reference point  (default : performance_cible.csv)
    - epsilon : (default = 0.1)

    point = m.nearest_alternative_to_a_reference_point(weight_file, performance_file, epsilon)


----------------------------------------------------------------------------------

Interactive method (simulating DM's requests with an aggregated function)

    /* Tchebycheff weights are by default in weight_file.csv */
    /* DM's linear agregation function is by default in DM_weights_file */

    m.start_exploration()



*****                                                                        *****
**********************************************************************************

**********************  PART 2 - IncrementalElicitation.py  **********************



Upload instance with the same format as "voitures.csv" :

    m = CSS_Solver('voitures.csv')


-------------------------------------------------------------------------------

Main elicitation method :
    /* DM's linear agregation function is by default in DM_weights_file */
    m.start()



*****                                                                        *****
**********************************************************************************

**************************  PART 3 - Knapsack_Model.py  **************************



Generate knapsack instances :
    - n : number of criteria
    - p : number of objects instance
    - filename : to save instance (default : knapsack_instance.csv)

    Knapsack_Model.generate_knapsack_instance(n,p,filename)


--------------------------------------------------------------------------------

Upload instance with the same format as "knapsack_instance.csv" :

    k = Knapsack_Model(filename_instance)


---------------------------------------------------------------------------------

Get nearest solution to ideal point :
    - epsilon : (default = 0.1)

    k.nearest_point_to_I(epsilon)


----------------------------------------------------------------------------------

Get nearest solution to a given reference point :
    - performance_file : data's reference point  (default : performance_cible.csv)
    - epsilon : (default = 0.1)

    point = k.nearest_alternative_to_a_reference_point(performance_file, epsilon)


----------------------------------------------------------------------------------

Interactive method (simulating DM's requests with an aggregated function)

    /* Tchebycheff weights are by default [1,1,.....,1] */
    /* DM's linear agregation function is by default in DM_weights_file_knapsack_2
        (there is alson DM_weights_file_knapsack_5.csv example for 5 criteria)*/

    k.start_exploration()




*****                                                                        *****
**********************************************************************************