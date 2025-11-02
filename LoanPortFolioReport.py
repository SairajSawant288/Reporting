import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import os
import gc


class LoanPortfolioReportAutomationPandas:
    def __init__(self, data_directory: str):
        self.data_directory = Path(data_directory)
        self.logger = self.setup_logging()
        self.output_report = None

    def setup_logging(self):
        # Ensure logs folder exists
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # File path for log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join('logs', f'loan_portfolio_{timestamp}.log')
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),                # Console output
                logging.FileHandler(log_file, mode='a') # File output, append mode
            ],
            force=True  # ensures config is applied even if called multiple times
        )
        
        self.logger = logging.getLogger('LoanPortfolioAutomationPandas')
        return self.logger

    # def load_all_dataframes(self):
    #     self.logger.info("Loading all data CSV files into DataFrames")
    #     self.aum_month_report = pd.read_csv(self.data_directory / "aum_month_report.csv")
    #     self.sharing_ratio_master = pd.read_csv(self.data_directory / "sharing_ratio_master.csv")
    #     self.bookdbet_tagging = pd.read_csv(self.data_directory / "bookdbet_tagging.csv")
    #     self.bookdbet_tagging_product = pd.read_csv(self.data_directory / "bookdbet_tagging_product.csv")
    #     self.write_off_lans = pd.read_csv(self.data_directory / "write_off_lans.csv")
    #     self.morat_data = pd.read_csv(self.data_directory / "morat_data.csv")
    #     self.foreclosure_tagging = pd.read_csv(self.data_directory / "foreclosure_tagging.csv")
    #     self.cal_managed_da_ptc_master = pd.read_csv(self.data_directory / "CAL_MANAGED_DA_PTC_Master.csv")
    #     self.overdue_report = pd.read_csv(self.data_directory / "overdue_report.csv")
    #     self.portfolio_report_opening = pd.read_csv(self.data_directory / "PORTFOLIO_REPORT_OPENING.csv")
    #     self.repo_stock_report = pd.read_csv(self.data_directory / "Repo_Stock_report.csv")
    #     self.co_lending_rate_master_report = pd.read_csv(self.data_directory / "co_lending_rate_master_report.csv")


    def load_all_dataframes(self):
        self.logger.info("Loading all data CSV files into DataFrames")

        files = {
            "aum_month_report": "aum_month_report.csv",
            "sharing_ratio_master": "sharing_ratio_master.csv",
            "bookdbet_tagging": "bookdbet_tagging.csv",
            "bookdbet_tagging_product": "bookdbet_tagging_product.csv",
            "write_off_lans": "write_off_lans.csv",
            "morat_data": "morat_data.csv",
            "foreclosure_tagging": "foreclosure_tagging.csv",
            "cal_managed_da_ptc_master": "CAL_MANAGED_DA_PTC_Master.csv",
            "overdue_report": "overdue_report.csv",
            "portfolio_report_opening": "PORTFOLIO_REPORT_OPENING.csv",
            "repo_stock_report": "Repo_Stock_report.csv",
            "co_lending_rate_master_report": "co_lending_rate_master_report.csv"
        }

        for var_name, filename in files.items():
            file_path = self.data_directory / filename
            try:
                # Try reading CSV
                df = pd.read_csv(file_path)
                setattr(self, var_name, df)
                self.logger.info(f"✅ Loaded {filename} ({len(df)} rows)")
            except FileNotFoundError:
                # File missing → create empty DataFrame
                self.logger.warning(f"⚠️ File not found: {filename} — using empty DataFrame")
                setattr(self, var_name, pd.DataFrame())
            except Exception as e:
                # Any other issue → still create empty DataFrame
                self.logger.warning(f"⚠️ Error loading {filename}: {e} — using empty DataFrame")
                setattr(self, var_name, pd.DataFrame())

        # ✅ Summary log
        summary = {name: getattr(self, name).shape for name in files.keys()}
        self.logger.info(f"Data loading summary (rows, cols): {summary}")



    def create_output_report_pandas(self):
        self.logger.info("Creating output_report DataFrame in pandas")
        self.logger.info(self.bookdbet_tagging_product.shape)
        self.bookdbet_tagging_product.rename(columns={'LOAN ACCOUNT NO.':'LOAN_ACCOUNT_NO'},inplace=True)
        
        aum_df = self.aum_month_report[['LOAN_ACCOUNT_NO','BRANCH_NAME','DISBURSAL_DATE','MONTH_YEAR','FINANCIAL_YEAR','LOANTYPE','COLENDER','SUBROGATION_TAGGING','TENURE','IRR','FINANCE_AMOUNT','WEMI_FINANCE_AMT','PASSIVE_FINANCE_AMOUNT','GROSS_ADVANCE_INSTL_AMT','GROSS_BILLED_ADVANCE_INSTL_AMT','UNBILLED_PRINCIPAL_AMOUNT']]
        bookdbet_tagging_product_df = self.bookdbet_tagging_product[['LOAN_ACCOUNT_NO','PRODUCT']]

        df = pd.merge(aum_df,bookdbet_tagging_product_df,on='LOAN_ACCOUNT_NO', how='left')
        del bookdbet_tagging_product_df
        gc.collect()
        del aum_df
        gc.collect()
        
        # df.drop(columns=['PRODUCT_x','TYPE'],inplace=True)
        # df.rename(columns={'PRODUCT_y':'PRODUCT'},inplace=True)
        

        def classify_type(row):
            product = row['PRODUCT']
            if product in ('DDF','DDF-New','DDF-Used','DDF-E Auto','DDF-NEW'):
                return 'Wholesale'
            st = row.get('SUBROGATION_TAGGING', '')
            # bm = row.get('BUSINESSMODEL', '')
            co = row.get('COLENDER', '')
            # if st == 'ICICI Subrogation':
            #     return 'ICICI Subrogation'
            # if bm == 'ICICI':
            #     return 'ICICI Co-lending'
            if st == 'Muthoot FLDG':
                return 'Muthoot FLDG'
            if st == 'Vivriti FLDG':
                return 'Vivriti FLDG'
            if st == 'Poonawala FLDG':
                return 'Poonawala FLDG'
            if co == 'Greaves co-lending':
                return 'Greaves Co-lending'
            if co == 'Muthoot Capital Co-lending':
                return 'Muthoot Capital Co-lending'
            if co == 'Vivriti Capital Co-lending':
                return 'Vivriti Capital Co-lending'
            if co == 'Poonawala Fincorp Co-lending':
                return 'Poonawala Fincorp Co-lending'
            if product == 'EV Insti':
                return 'Institutional Underwriting - Loan'
            return 'Retail'
        
        df['TYPE'] = df.apply(classify_type, axis=1)
        

        df['unbilled_adv_emi'] = df.get('GROSS_ADVANCE_INSTL_AMT', 0).fillna(0) - df.get('GROSS_BILLED_ADVANCE_INSTL_AMT', 0).fillna(0)
        df['total_osp_from_finone'] = df.get('UNBILLED_PRINCIPAL_AMOUNT', 0)
        df.drop(columns=['GROSS_ADVANCE_INSTL_AMT','GROSS_BILLED_ADVANCE_INSTL_AMT','UNBILLED_PRINCIPAL_AMOUNT'],inplace=True)
        

        
        # Join tagging
        tagging = self.bookdbet_tagging.rename(columns={'LAN': 'LOAN_ACCOUNT_NO','DA / PTC TRANCH WISE - BOOKDEBT':'DA_PTC_TRANCH_WISE_BOOKDEBT','DA / PTC TRANCH WISE - AUM':'DA_PTC_TRANCH_WISE_AUM','Managed DA / PTC':'MANAGED_DA_PTC'})[
            ['LOAN_ACCOUNT_NO', 'DA_PTC_TRANCH_WISE_BOOKDEBT', 'DA_PTC_TRANCH_WISE_AUM', 'MANAGED_DA_PTC']]
        df = pd.merge(df, tagging, on='LOAN_ACCOUNT_NO', how='left')
        del tagging
        gc.collect()

        
        # Join write-off info
        
        write_off_renamed = self.write_off_lans.rename(columns={'LAN': 'LOAN_ACCOUNT_NO', 'WRITE OFF': 'write_off_status', 'Month': 'write_off_date'})
        # Remove rows where ALL THREE columns are null
        write_off_renamed = write_off_renamed.dropna(subset=['LOAN_ACCOUNT_NO', 'write_off_status', 'write_off_date'], how='all')

        write_off_renamed_ = write_off_renamed[['LOAN_ACCOUNT_NO', 'write_off_status', 'write_off_date']]
        print("write_off_renamed")
        print(write_off_renamed_.shape)
        print("final df")
        print(df.shape)
        df = pd.merge(df, write_off_renamed_, on='LOAN_ACCOUNT_NO', how='left')
        print("AUM Book Dept")
        print(df.shape)
        df['write_off_status'] = df['write_off_status'].fillna('NO')
        # self.logger.info("Generated df",df.shape)
        
        self.output_report = df
        self.logger.info("output_report DataFrame created")

    def process_full_business_logic_pandas(self):
        df = self.output_report
        srm = self.sharing_ratio_master
        amr = self.aum_month_report
        morat = self.morat_data

        df['osp_final'] = np.where(df['MANAGED_DA_PTC'] == 'Managed - ARC', 0, df['total_osp_from_finone'])
        df['osp_final_excluding_write_off'] = df['osp_final'].copy()
        df.loc[df['write_off_status'] == 'YES', 'osp_final_excluding_write_off'] = 0

        overdue_dict = amr.set_index('LOAN_ACCOUNT_NO')['OVERDUE_PRIN_AMT'].to_dict()
        df['OVERDUE_PRIN_AMT'] = df['LOAN_ACCOUNT_NO'].map(overdue_dict)
        df['overdue_prin_amt_excluding_write_off'] = df['OVERDUE_PRIN_AMT'].copy()
        df.loc[df['MANAGED_DA_PTC'] == 'Managed - ARC', 'overdue_prin_amt_excluding_write_off'] = 0
        df.loc[df['write_off_status'] == 'YES', 'overdue_prin_amt_excluding_write_off'] = 0
        df['overdue_prin_amt_excluding_write_off'] = np.where(df['overdue_prin_amt_excluding_write_off'] < 0,0,df['overdue_prin_amt_excluding_write_off'])

        srm.rename(columns={'DA / PTC Tranch wise - AUM':'DA_PTC_TRANCH_WISE_AUM','Own Share':'OWN_SHARE'},inplace=True)
        def calc_principal_od_own(row):
            # Lookup OWN_SHARE for the row’s type and DA_PTC_TRANCH_WISE_AUM
            type_match = srm.loc[srm['DA_PTC_TRANCH_WISE_AUM'] == row['TYPE'], 'OWN_SHARE']
            daptc_match = srm.loc[srm['DA_PTC_TRANCH_WISE_AUM'] == row['DA_PTC_TRANCH_WISE_AUM'], 'OWN_SHARE']

            # If lookup yields no rows, treat share as 0
            type_share = float(type_match.iloc[0]) if not type_match.empty and pd.notna(type_match.iloc[0]) else 0.0
            daptc_share = float(daptc_match.iloc[0]) if not daptc_match.empty and pd.notna(daptc_match.iloc[0]) else 0.0

            overdue = row.get('overdue_prin_amt_excluding_write_off', 0.0) or 0.0


            # If overdue is missing or zero, result is zero
            if overdue <= 0:
                return 0.0

            # Compute combined percentage
            type_per = type_share / 100.0
            daptc_per = daptc_share / 100.0


            # If either percentage is zero, result is zero
            if type_per == 0.0 and daptc_per == 0.0:
                return overdue

            # Final calculation
            result = 0
            if daptc_per > 0 and type_per > 0:
                result = round(overdue * type_per * daptc_per,2)
                return max(result, 0.0)
            if type_per > 0:
                result = round(overdue * type_per,2)
                return max(result, 0.0)
            if daptc_per > 0:
                result = round(overdue * daptc_per,2)
                return max(result, 0.0)

            # Ensure non-negative
            

        df['principal_od_own_from_par'] = df.apply(calc_principal_od_own, axis=1)
        
        morat_dict = morat.set_index('LOAN_ACCOUNT_NO')['MORAT'].to_dict()
        df['morat_principal_from_tab'] = df['LOAN_ACCOUNT_NO'].map(morat_dict).fillna(0)
        df.loc[df['write_off_status'] == 'YES', 'morat_principal_from_tab'] = 0
        df.loc[df['MANAGED_DA_PTC'] == 'Managed - ARC', 'morat_principal_from_tab'] = 0
        
        df['aum_including_written_off'] = df['osp_final_excluding_write_off'].fillna(0) + df['principal_od_own_from_par'].fillna(0) + df['morat_principal_from_tab'].fillna(0)

        df['write_off_amt'] = 0
        df['aum_excluding_written_off'] = df['aum_including_written_off'].copy()
        # df.loc[df['write_off_amt'] > 0, 'aum_excluding_written_off'] = 0
        df.loc[df['write_off_status'] == 'YES', 'aum_excluding_written_off'] = 0
        df.loc[df['MANAGED_DA_PTC'] == 'Managed - ARC', 'aum_excluding_written_off'] = 0

        # Additional logic similar to above SQL updates can be added here
        # Vishal concern 
        df['PRINCIPAL_OD_Managed'] = df['overdue_prin_amt_excluding_write_off'] - df['principal_od_own_from_par']

        df['AUM_including_advance_EMI'] = df['aum_excluding_written_off'] + df['PRINCIPAL_OD_Managed']
        df.loc[df['write_off_status'] == 'YES', 'AUM_including_advance_EMI'] = 0
        df.loc[df['MANAGED_DA_PTC'] == 'Managed - ARC', 'AUM_including_advance_EMI'] = 0

        df['ADVANCE_EMI'] = df['unbilled_adv_emi'].copy()
        df.loc[df['write_off_status'] == 'YES', 'ADVANCE_EMI'] = 0
        df.loc[df['MANAGED_DA_PTC'] == 'Managed - ARC', 'ADVANCE_EMI'] = 0

        df['Final_AUM'] = df['AUM_including_advance_EMI'] - df['ADVANCE_EMI']
        df.loc[df['write_off_status'] == 'YES', 'Final_AUM'] = 0
        df.loc[df['MANAGED_DA_PTC'] == 'Managed - ARC', 'Final_AUM'] = 0
        df['Final_AUM'] = np.where(df['Final_AUM'] < 0,0,df['Final_AUM'])

        df['OSP_Final_Min_ADVANCE_EMI'] = df['osp_final_excluding_write_off'] - df['ADVANCE_EMI']
        df.loc[df['write_off_status'] == 'YES', 'OSP_Final_Min_ADVANCE_EMI'] = 0
        df.loc[df['MANAGED_DA_PTC'] == 'Managed - ARC', 'OSP_Final_Min_ADVANCE_EMI'] = 0

        # Future owned shares
        def calc_future_owned_shares(row):
            
            # Lookup OWN_SHARE for the row’s type and DA_PTC_TRANCH_WISE_AUM
            type_match = srm.loc[srm['DA_PTC_TRANCH_WISE_AUM'] == row['TYPE'], 'OWN_SHARE']
            daptc_match = srm.loc[srm['DA_PTC_TRANCH_WISE_AUM'] == row['DA_PTC_TRANCH_WISE_AUM'], 'OWN_SHARE']
            
            # If lookup yields no rows, treat share as 0
            type_share = float(type_match.iloc[0]) if not type_match.empty and pd.notna(type_match.iloc[0]) else 0.0
            daptc_share = float(daptc_match.iloc[0]) if not daptc_match.empty and pd.notna(daptc_match.iloc[0]) else 0.0

            
            overdue = row.get('OSP_Final_Min_ADVANCE_EMI', 0.0) or 0.0

            
            # If overdue is missing or zero, result is zero
            if overdue <= 0:
                return 0.0

            # Compute combined percentage
            type_per = type_share / 100.0
            daptc_per = daptc_share / 100.0

           
            # If either percentage is zero, result is zero
            if type_per == 0.0 and daptc_per == 0.0:
                return overdue

            # Final calculation
            result = 0
            if daptc_per > 0 and type_per > 0:
                result = round(overdue * type_per * daptc_per,2)
                return max(result, 0.0)
            if type_per > 0:
                result = round(overdue * type_per,2)
                return max(result, 0.0)
            if daptc_per > 0:
                result = round(overdue * daptc_per,2)
                return max(result, 0.0)
            

        # df_ = df[(df['LOAN_ACCOUNT_NO'] == 'MCOMCN000005240049') | (df['LOAN_ACCOUNT_NO'] == 'JANN2W000005034582') | (df['LOAN_ACCOUNT_NO'] == 'MUMVCU000005283597')]
        df['FUTURE_OWNED_SHARE'] = df.apply(calc_future_owned_shares,axis=1)
        # df_ = df[(df['LOAN_ACCOUNT_NO'] == 'MCOMCN000005240049') | (df['LOAN_ACCOUNT_NO'] == 'JANN2W000005034582') | (df['LOAN_ACCOUNT_NO'] == 'MUMVCU000005283597')]

        df['FUTURE_MANAGED_SHARE'] = df['OSP_Final_Min_ADVANCE_EMI'] - df['FUTURE_OWNED_SHARE']

        # MORAT_OWN_AMOUNT
        def calc_morat_own_amount(row):
            
            # Lookup OWN_SHARE for the row’s type and DA_PTC_TRANCH_WISE_AUM
            type_match = srm.loc[srm['DA_PTC_TRANCH_WISE_AUM'] == row['TYPE'], 'OWN_SHARE']
            daptc_match = srm.loc[srm['DA_PTC_TRANCH_WISE_AUM'] == row['DA_PTC_TRANCH_WISE_AUM'], 'OWN_SHARE']
            
            # If lookup yields no rows, treat share as 0
            type_share = float(type_match.iloc[0]) if not type_match.empty and pd.notna(type_match.iloc[0]) else 0.0
            daptc_share = float(daptc_match.iloc[0]) if not daptc_match.empty and pd.notna(daptc_match.iloc[0]) else 0.0

            
            overdue = row.get('morat_principal_from_tab', 0.0) or 0.0

            
            # If overdue is missing or zero, result is zero
            if overdue <= 0:
                return 0.0

            # Compute combined percentage
            type_per = type_share / 100.0
            daptc_per = daptc_share / 100.0

            
            # If either percentage is zero, result is zero
            if type_per == 0.0 and daptc_per == 0.0:
                # print("Inside Overdue")
                return overdue

            # Final calculation
            result = 0
            if daptc_per > 0 and type_per > 0:
                result = round(overdue * type_per * daptc_per,2)
                return max(result, 0.0)
            if type_per > 0:
                result = round(overdue * type_per,2)
                return max(result, 0.0)
            if daptc_per > 0:
                result = round(overdue * daptc_per,2)
                return max(result, 0.0)
            

        df['MORAT_OWN_AMOUNT'] = df.apply(calc_morat_own_amount,axis=1)

        df['MORAT_MANAGED_AMOUNT'] = df['morat_principal_from_tab'] - df['MORAT_OWN_AMOUNT']

        df['TOTAL_AUM_OWNED_SHARE'] = df['principal_od_own_from_par'] + df['FUTURE_OWNED_SHARE'] + df['MORAT_OWN_AMOUNT']

        df['TOTAL_AUM_MANAGED_SHARE'] = df['PRINCIPAL_OD_Managed'] + df['FUTURE_MANAGED_SHARE'] + df['MORAT_MANAGED_AMOUNT']
        df = df[df['write_off_status'] != 'YES']

        # df.columns = df.columns.str.upper()
        self.foreclosure_tagging.rename(columns={'LANs':'LOAN_ACCOUNT_NO'},inplace=True)
        df = df[~df['LOAN_ACCOUNT_NO'].isin(self.foreclosure_tagging['LOAN_ACCOUNT_NO'])]
        # df.to_csv("loan_portfolio_report.csv",index=False)
        # print("report completed")
        # 15th changes
        overdue_report_df = self.overdue_report[['AGREEMENT_NO','EMI_OVERDUE','DPD']]
        df = pd.merge(df,overdue_report_df , left_on='LOAN_ACCOUNT_NO',right_on='AGREEMENT_NO', how='left')
        df['CLOSING_EMI_OVERDUE'] = df['EMI_OVERDUE'].copy()
        df.drop(columns=['AGREEMENT_NO','EMI_OVERDUE'],inplace=True)

        #Testing Remaining
        df['MONTH_END_DPD'] = np.where(df['CLOSING_EMI_OVERDUE'] > 0,df['DPD'],'')

        portfolio_report_opening_df = self.portfolio_report_opening[['LOAN_ACCOUNT_NO','ASSET_CLASSIFICATION','MATURITYDATE']]

        df = pd.merge(df,portfolio_report_opening_df , on='LOAN_ACCOUNT_NO', how='left')
        

        df['ASSET_CLASSIFICATION_FINONE'] = np.where(df['CLOSING_EMI_OVERDUE'] > 0,df['ASSET_CLASSIFICATION'],'')
        df.drop(columns=['ASSET_CLASSIFICATION'],inplace=True)
        df['DATE_OF_MATURITY'] = df['MATURITYDATE'].copy()

        repo_stock_report_df = self.repo_stock_report[['Loan Number','Vehicle Status']]
        repo_stock_report_df.rename(columns={'Loan Number':'LOAN_ACCOUNT_NO','Vehicle Status':'REPO_STOCK_TAGGING'},inplace=True)

        df = pd.merge(df,repo_stock_report_df,on='LOAN_ACCOUNT_NO',how='left')

        def sma_calculate(row):
            try:
                if float(row.get('CLOSING_EMI_OVERDUE', 0)) <= 0:
                    return ""
            except (ValueError, TypeError):
                return ""

            mapping = {
                'STANDARD': 'SMA-0',
                'SMA - 0: 1 to 30 Days': 'SMA-0',
                'SMA -1: More than 30 days to 60 days': 'SMA-1',
                'SMA - 2: More than 60 days to 90 days': 'SMA-2'
            }

            return mapping.get(row.get('ASSET_CLASSIFICATION_FINONE'), 'NPA')

        df['SMA'] = df.apply(sma_calculate,axis=1)

        def asset_class_final_calculate(row):
            sma = row.get('SMA', '')
            repo_stock_tagging = row.get('REPO_STOCK_TAGGING', '')
            asset_class = row.get('ASSET_CLASSIFICATION_FINONE', '')
            closing_overdue = row.get('CLOSING_EMI_OVERDUE', 0)

            # Match SQL WHERE clause
            if closing_overdue <= 0:
                return ""

            if sma == 'NPA' and repo_stock_tagging == 'REPO STOCK':
                return '20% Repo Stock - NPA'
            elif repo_stock_tagging == 'REPO STOCK':
                return '20% Repo Stock'
            elif asset_class in [
                'STANDARD',
                'SMA - 0: 1 to 30 Days',
                'SMA -1: More than 30 days to 60 days',
                'SMA - 2: More than 60 days to 90 days'
            ]:
                return 'STANDARD'
            else:
                return 'SUBSTANDARD'


        df['ASSET_CLASS_FINAL'] = df.apply(asset_class_final_calculate,axis=1)
        df.to_csv('test.csv',index=False)

        def bucket_category(row):
            closing_overdue = row.get('CLOSING_EMI_OVERDUE', 0)
            month_end_dpd = row.get('MONTH_END_DPD')

            # Return empty if no overdue
            if closing_overdue <= 0 or month_end_dpd == '':
                return ''

            # Apply DPD-based bucketing
            if eval(month_end_dpd) == 0:
                return 'Current'
            elif 1 <= eval(month_end_dpd) < 31:
                return '1-30'
            elif 30 < eval(month_end_dpd) < 61:
                return '31-60'
            elif 60 < eval(month_end_dpd) < 91:
                return '61-90'
            elif 90 < eval(month_end_dpd) < 121:
                return '91-120'
            elif 120 < eval(month_end_dpd) < 151:
                return '121-150'
            elif 150 < eval(month_end_dpd) < 181:
                return '151-180'
            elif 180 < eval(month_end_dpd) < 211:
                return '181-210'
            elif 210 < eval(month_end_dpd) < 241:
                return '211-240'
            elif 240 < eval(month_end_dpd) < 271:
                return '241-270'
            elif 270 < eval(month_end_dpd) < 301:
                return '271-300'
            elif 300 < eval(month_end_dpd) < 331:
                return '301-330'
            elif 330 < eval(month_end_dpd) < 366:
                return '331-365'
            elif 365 < eval(month_end_dpd) < 396:
                return '366-395'
            elif 395 < eval(month_end_dpd) < 426:
                return '396-425'
            elif eval(month_end_dpd) > 425:
                return '425+'
            else:
                return ''

        df['BUCKET_CATEGORY'] = df.apply(bucket_category,axis=1)

        def calc_wemi_adv_emi(row):
            # Lookup OWN_SHARE for the row’s type and DA_PTC_TRANCH_WISE_AUM
            if row['MANAGED_DA_PTC'] == 'Managed - ARC':
                return 0
            type_match = srm.loc[srm['DA_PTC_TRANCH_WISE_AUM'] == row['TYPE'], 'OWN_SHARE']
            daptc_match = srm.loc[srm['DA_PTC_TRANCH_WISE_AUM'] == row['DA_PTC_TRANCH_WISE_AUM'], 'OWN_SHARE']

            # If lookup yields no rows, treat share as 0
            type_share = float(type_match.iloc[0]) if not type_match.empty and pd.notna(type_match.iloc[0]) else 0.0
            daptc_share = float(daptc_match.iloc[0]) if not daptc_match.empty and pd.notna(daptc_match.iloc[0]) else 0.0

            overdue = row.get('ADVANCE_EMI', 0.0) or 0.0


            # If overdue is missing or zero, result is zero
            if overdue <= 0:
                return 0.0

            # Compute combined percentage
            type_per = type_share / 100.0
            daptc_per = daptc_share / 100.0


            # If either percentage is zero, result is zero
            if type_per == 0.0 and daptc_per == 0.0:
                return overdue

            # Final calculation
            result = 0
            if daptc_per > 0 and type_per > 0:
                result = round(overdue * type_per * daptc_per,2)
                return max(result, 0.0)
            if type_per > 0:
                result = round(overdue * type_per,2)
                return max(result, 0.0)
            if daptc_per > 0:
                result = round(overdue * daptc_per,2)
                return max(result, 0.0)
            
        df['WEMI_ADV_EMI'] = df.apply(calc_wemi_adv_emi,axis=1)

        df['MANAGED_ADV_EMI'] = df['ADVANCE_EMI'] - df['WEMI_ADV_EMI']
        df.loc[df['MANAGED_DA_PTC'] == 'Managed - ARC', 'MANAGED_ADV_EMI'] = 0

        df['TOTAL_INTEREST_OD_OWN_MANAGED'] = df['CLOSING_EMI_OVERDUE'] - df['overdue_prin_amt_excluding_write_off']
        df.loc[df['MANAGED_DA_PTC'] == 'Managed - ARC', 'TOTAL_INTEREST_OD_OWN_MANAGED'] = 0

        def calc_provision_percent(row):
            asset_class = str(row.get('ASSET_CLASS_FINAL', '')).strip().upper()

            if asset_class == 'LOSS':
                return 100
            elif asset_class == 'LOSS-MORAT':
                return 100
            elif asset_class == '20% Repo Stock':
                return 20
            elif asset_class == '20% Repo Stock - NPA':
                return 20
            elif asset_class == 'SUBSTANDARD':
                return 10
            elif asset_class == 'DOUBTFUL HYD CASES':
                return 100
            elif asset_class == 'STANDARD RESCHEDULED EXPIRED':
                return 10
            else:
                return 0

        df['PROVISION_PERCENT'] = df.apply(calc_provision_percent,axis=1)

        # def calc_cal_managed_da_ptc(row):
        #     """
        #     Matches CAL_MANAGED_DA_PTC from master_df based on TYPE logic:
        #     - If detailed type contains 'co-lending' → match with master TYPE='Co-Lending'
        #     - If detailed type contains 'fldg' → match with master TYPE='FLDG'
        #     - Else → match with master TYPE='Other'
        #     Falls back to AUM-only match if no TYPE match found.
        #     """
        #     row_type = str(row.get('TYPE', '')).lower()
        #     da_ptc_tranch_wise_aum = row.get('DA_PTC_TRANCH_WISE_AUM')

        #     # 🧠 Determine the normalized master type
        #     if 'co-lending' in row_type:
        #         master_type = 'Co-Lending'
        #     elif 'fldg' in row_type:
        #         master_type = 'FLDG'
        #     else:
        #         master_type = 'Other'

        #     # Step 1️⃣ Try to match both AUM and master type
        #     matches = self.cal_managed_da_ptc_master[
        #         (self.cal_managed_da_ptc_master['TYPE'].str.lower() == master_type.lower()) &
        #         (self.cal_managed_da_ptc_master['DA_PTC_TRANCH_WISE_AUM'] == da_ptc_tranch_wise_aum)
        #     ]

        #     # Step 2️⃣ Fallback: match only by DA_PTC_TRANCH_WISE_AUM
        #     if matches.empty:
        #         if 
        #         matches = self.cal_managed_da_ptc_master[
        #             self.cal_managed_da_ptc_master['DA_PTC_TRANCH_WISE_AUM'] == da_ptc_tranch_wise_aum
        #         ]
            
            

        #     # Step 3️⃣ Return matching value or None
        #     if not matches.empty:
        #         return matches.iloc[0]['CAL_MANAGED_DA_PTC']
        #     else:
        #         return None

        def calc_cal_managed_da_ptc(self, row):
            """
            Matches CAL_MANAGED_DA_PTC based on priority:
            1) Match both TYPE and DA_PTC_TRANCH_WISE_AUM (if both exist)
            2) If only TYPE exists → match TYPE
            3) If only AUM exists → match DA_PTC_TRANCH_WISE_AUM
            4) Else return None
            """

            row_type = str(row.get('TYPE', '')).lower()
            da_ptc_tranch_wise_aum = row.get('DA_PTC_TRANCH_WISE_AUM')

            # Determine normalized type
            if 'co-lending' in row_type:
                master_type = 'Co-Lending'
            elif 'fldg' in row_type:
                master_type = 'FLDG'
            else:
                master_type = 'Other'

            master_df = self.cal_managed_da_ptc_master

            # ✅ Case 1: Both TYPE & AUM available
            if master_type and pd.notna(da_ptc_tranch_wise_aum):
                matches = master_df[
                    (master_df['TYPE'].str.lower() == master_type.lower()) &
                    (master_df['DA_PTC_TRANCH_WISE_AUM'] == da_ptc_tranch_wise_aum)
                ]
                if not matches.empty:
                    return matches.iloc[0]['CAL_MANAGED_DA_PTC']

            # ✅ Case 2: Only TYPE present
            if master_type:
                matches = master_df[
                    master_df['TYPE'].str.lower() == master_type.lower()
                ]
                if not matches.empty:
                    return matches.iloc[0]['CAL_MANAGED_DA_PTC']

            # ✅ Case 3: Only AUM present
            if pd.notna(da_ptc_tranch_wise_aum):
                matches = master_df[
                    master_df['DA_PTC_TRANCH_WISE_AUM'] == da_ptc_tranch_wise_aum
                ]
                if not matches.empty:
                    return matches.iloc[0]['CAL_MANAGED_DA_PTC']

            # ❌ No match found
            return None




        df['CAL_MANAGED_DA_PTC'] = df.apply(calc_cal_managed_da_ptc,axis=1)

        co_lending_rate_master_report_df = self.co_lending_rate_master_report[['LOAN_ACCOUNT_NO','CO_LENDING_INTREST_RATE']]

        df = pd.merge(df,co_lending_rate_master_report_df,on='LOAN_ACCOUNT_NO',how='left')

        # df['CO_LENDING_INTREST_RATE'] = 
        df.columns = df.columns.str.upper()

        

        self.output_report = df
        self.logger.info("Business logic processing completed")

    def export_output_report(self, report_directory='loan_portfolio_report.csv'):
        self.output_report.to_csv(report_directory, index=False)
        self.logger.info(f"Exported output_report as {report_directory}")




if __name__ == "__main__":
    data_dir = "/Users/sairajsawant/Desktop/work/Vishal/code/data" # Update this path accordingly

    automation = LoanPortfolioReportAutomationPandas(data_dir)
    automation.load_all_dataframes()
    automation.create_output_report_pandas()
    automation.process_full_business_logic_pandas()
    automation.export_output_report("/Users/sairajsawant/Desktop/work/Vishal/code/Generated_Report/loan_portfolio_report.csv")

    print("Loan portfolio report generated successfully.")
