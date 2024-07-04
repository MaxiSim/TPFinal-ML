from scripts.feature_ing.main_clean_eng_test import process_test_data
from scripts.models.predict import predict

def main ():
    
    test_file =  'data/pf_suvs_test_ids_i302_1s2024.csv'
    process_test_data(test_file)
    print('Test data processed')
    predict('xgboost', 'xg_opt_over')
    print('Predictions saved to results/predictions/predictions_Simian_Manzano.csv')
    
if __name__ == "__main__":
    main()
    