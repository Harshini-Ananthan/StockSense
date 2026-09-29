import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

def main():
    # Set random seed
    np.random.seed(42)
    random.seed(42)

    print("Generating Synthetic Data for STOCKSENSE...")
    
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Generate Stores
    stores_data = [
        {"store_id": "S01", "city": "Coimbatore", "store_type": "Supermarket", "floor_area_sqft": 8500, "avg_daily_customers": 1250, "region": "West"},
        {"store_id": "S02", "city": "Chennai", "store_type": "Hypermarket", "floor_area_sqft": 18000, "avg_daily_customers": 2850, "region": "North"},
        {"store_id": "S03", "city": "Madurai", "store_type": "Express", "floor_area_sqft": 4200, "avg_daily_customers": 720, "region": "South"},
        {"store_id": "S04", "city": "Salem", "store_type": "Supermarket", "floor_area_sqft": 7600, "avg_daily_customers": 1080, "region": "Central"}
    ]
    df_stores = pd.DataFrame(stores_data)
    
    # 2. Generate Products
    products_data = [
        {"product_id": "P101", "category": "Dairy", "sub_category": "Milk", "brand": "Aavin", "mrp": 60, "cost_price": 52, "shelf_life_days": 3, "supplier_id": "SUP01", "base_demand": 50},
        {"product_id": "P102", "category": "Dairy", "sub_category": "Cheese", "brand": "MilkyWay", "mrp": 150, "cost_price": 110, "shelf_life_days": 30, "supplier_id": "SUP01", "base_demand": 15},
        {"product_id": "P201", "category": "Personal Care", "sub_category": "Soap", "brand": "GlowCare", "mrp": 45, "cost_price": 30, "shelf_life_days": 730, "supplier_id": "SUP02", "base_demand": 25},
        {"product_id": "P205", "category": "Personal Care", "sub_category": "Shampoo", "brand": "GlowCare", "mrp": 120, "cost_price": 79, "shelf_life_days": 730, "supplier_id": "SUP08", "base_demand": 20},
        {"product_id": "P330", "category": "Beverages", "sub_category": "Soft Drink", "brand": "FizzUp", "mrp": 50, "cost_price": 31, "shelf_life_days": 180, "supplier_id": "SUP04", "base_demand": 40},
        {"product_id": "P331", "category": "beverage", "sub_category": "Juice", "brand": "NatureFresh", "mrp": 90, "cost_price": 60, "shelf_life_days": 120, "supplier_id": "SUP04", "base_demand": 30},
        {"product_id": "P442", "category": "Snacks", "sub_category": "Biscuits", "brand": "Crispo", "mrp": 35, "cost_price": 22, "shelf_life_days": 240, "supplier_id": "SUP06", "base_demand": 60},
        {"product_id": "P443", "category": "Snacks", "sub_category": "Chips", "brand": "Crunchy", "mrp": 20, "cost_price": 12, "shelf_life_days": 180, "supplier_id": "SUP06", "base_demand": 80},
        {"product_id": "P501", "category": "Groceries", "sub_category": "Rice", "brand": "FarmGold", "mrp": 600, "cost_price": 500, "shelf_life_days": 365, "supplier_id": "SUP07", "base_demand": 10},
        {"product_id": "P502", "category": "Groceries", "sub_category": "Dal", "brand": "FarmGold", "mrp": 120, "cost_price": 95, "shelf_life_days": 365, "supplier_id": "SUP07", "base_demand": 20},
        {"product_id": "P601", "category": "Household", "sub_category": "Cleaner", "brand": "Shine", "mrp": 99, "cost_price": 70, "shelf_life_days": 730, "supplier_id": "SUP03", "base_demand": 15},
        {"product_id": "P701", "category": "Frozen", "sub_category": "Peas", "brand": "IceFarm", "mrp": 80, "cost_price": 55, "shelf_life_days": 180, "supplier_id": "SUP05", "base_demand": 12},
        {"product_id": "P702", "category": "Frozen", "sub_category": "Ice Cream", "brand": "ChillZ", "mrp": 250, "cost_price": 180, "shelf_life_days": 90, "supplier_id": "SUP05", "base_demand": 25},
        {"product_id": "P999", "category": "BEVERAGES", "sub_category": "Energy Drink", "brand": "Zap", "mrp": 110, "cost_price": 75, "shelf_life_days": 180, "supplier_id": "SUP04", "base_demand": 20}, # Sparse product
    ]
    df_products = pd.DataFrame(products_data)
    
    # 3. External Factors
    start_date = datetime(2026, 5, 1)
    end_date = datetime(2026, 8, 31)
    date_list = [start_date + timedelta(days=x) for x in range((end_date-start_date).days + 1)]
    
    cities = df_stores['city'].unique()
    ext_factors = []
    
    for dt in date_list:
        dt_str = dt.strftime('%Y-%m-%d')
        is_weekend = 1 if dt.weekday() >= 5 else 0
        is_holiday = 1 if dt_str in ['2026-05-01', '2026-08-15'] else 0
        is_festival = 1 if dt_str in ['2026-08-25', '2026-08-26'] else 0 # Mock festival
        
        for city in cities:
            temp_c = round(np.random.normal(32, 3), 1)
            rain_mm = round(max(0, np.random.normal(0, 5)), 1)
            local_event = 1 if np.random.rand() < 0.05 else 0
            
            ext_factors.append({
                "date": dt_str,
                "city": city,
                "temp_c": temp_c,
                "rain_mm": rain_mm,
                "holiday": is_holiday,
                "festival": is_festival,
                "weekend": is_weekend,
                "local_event": local_event
            })
    df_ext = pd.DataFrame(ext_factors)
    
    # Inject Missing values in external_factors (temp_c)
    missing_idx = df_ext.sample(frac=0.03).index
    df_ext.loc[missing_idx, 'temp_c'] = np.nan
    
    # 4. Generate Inventory and Transactions
    inventory = []
    transactions = []
    
    # Track inventory state: {store: {product: current_stock}}
    current_stock = {}
    for s in df_stores['store_id']:
        current_stock[s] = {}
        for p in df_products['product_id']:
            current_stock[s][p] = np.random.randint(50, 200)
    
    transaction_id_counter = 10000
    
    for dt in date_list:
        dt_str = dt.strftime('%Y-%m-%d')
        is_wknd = 1 if dt.weekday() >= 5 else 0
        is_fest = 1 if dt_str in ['2026-08-25', '2026-08-26'] else 0
        
        for idx, store_row in df_stores.iterrows():
            s_id = store_row['store_id']
            store_multiplier = store_row['avg_daily_customers'] / 1000.0
            
            for p_idx, prod_row in df_products.iterrows():
                p_id = prod_row['product_id']
                
                # Sparse history logic for P999 (only available in last 10 days)
                if p_id == "P999" and dt < (end_date - timedelta(days=10)):
                    continue
                    
                base_d = prod_row['base_demand']
                
                # Promotions (10% chance)
                is_promo = 1 if np.random.rand() < 0.1 else 0
                promo_multiplier = 1.5 if is_promo else 1.0
                
                # Weekend / Festival effect
                wknd_multiplier = 1.3 if is_wknd else 1.0
                fest_multiplier = 1.8 if is_fest else 1.0
                
                demand = base_d * store_multiplier * promo_multiplier * wknd_multiplier * fest_multiplier
                demand = int(np.random.poisson(demand))
                
                # Check stock and fulfill
                opening = current_stock[s_id][p_id]
                
                # Replenishment
                reorder_lvl = int(base_d * store_multiplier * 3)
                target_stock = int(base_d * store_multiplier * 10)
                received = 0
                if opening <= reorder_lvl:
                    received = target_stock - opening
                
                avail = opening + received
                sold = min(demand, avail)
                closing = avail - sold
                current_stock[s_id][p_id] = closing
                
                lead_days = np.random.randint(1, 5)
                
                inv_row = {
                    "date": dt_str,
                    "store": s_id,
                    "product": p_id,
                    "opening": opening,
                    "received": received,
                    "sold": sold,
                    "closing": closing,
                    "reorder_lvl": reorder_lvl,
                    "lead_days": lead_days
                }
                inventory.append(inv_row)
                
                # Generate Transactions
                if sold > 0:
                    remaining_sold = sold
                    while remaining_sold > 0:
                        txn_qty = min(remaining_sold, np.random.randint(1, 5))
                        if txn_qty == 0:
                            break
                        remaining_sold -= txn_qty
                        
                        disc_pct = float(np.random.randint(5, 20)) if is_promo else 0.0
                        sell_price = prod_row['mrp'] * (1 - disc_pct/100)
                        
                        txn = {
                            "transaction_id": f"T{transaction_id_counter}",
                            "date": dt_str,
                            "store_id": s_id,
                            "product_id": p_id,
                            "quantity": txn_qty,
                            "selling_price": round(sell_price, 2),
                            "discount_pct": disc_pct,
                            "promotion_flag": is_promo,
                            "customer_id": f"C{np.random.randint(1000, 9999)}",
                            "payment_mode": np.random.choice(["UPI", "Card", "Cash"], p=[0.5, 0.3, 0.2]),
                            "hour": int(np.random.normal(14, 3) % 24)
                        }
                        transactions.append(txn)
                        transaction_id_counter += 1

    df_inv = pd.DataFrame(inventory)
    df_txn = pd.DataFrame(transactions)
    
    # 5. Inject Error in transactions (duplicate rows, impossible quantity)
    # Duplicate some rows
    dup_samples = df_txn.sample(n=10, random_state=1)
    df_txn = pd.concat([df_txn, dup_samples], ignore_index=True)
    
    # Impossible quantity
    neg_qty_idx = df_txn.sample(n=5, random_state=2).index
    df_txn.loc[neg_qty_idx, 'quantity'] = -4
    
    # 6. Inject Error in inventory (arithmetic mismatch)
    mismatch_idx = df_inv.sample(n=20, random_state=3).index
    df_inv.loc[mismatch_idx, 'closing'] = df_inv.loc[mismatch_idx, 'closing'] + np.random.choice([-10, 10, -5, 5], size=20)
    
    # Remove 'base_demand' from products
    df_products.drop(columns=['base_demand'], inplace=True)
    
    # Save to CSV
    df_txn.to_csv(os.path.join(out_dir, 'transactions.csv'), index=False)
    df_products.to_csv(os.path.join(out_dir, 'products.csv'), index=False)
    df_stores.to_csv(os.path.join(out_dir, 'stores.csv'), index=False)
    df_inv.to_csv(os.path.join(out_dir, 'inventory.csv'), index=False)
    df_ext.to_csv(os.path.join(out_dir, 'external_factors.csv'), index=False)
    
    print("Data Generation Complete.")
    print(f"Transactions: {len(df_txn)}")
    print(f"Products: {len(df_products)}")
    print(f"Stores: {len(df_stores)}")
    print(f"Inventory: {len(df_inv)}")
    print(f"External Factors: {len(df_ext)}")
    print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("Injected Traps:")
    print("- Missing values in external_factors (temp_c): ~3%")
    print("- Category inconsistency in products: ('beverage', 'BEVERAGES')")
    print("- Sparse history for product: P999")
    print("- Duplicate rows in transactions: 10 rows")
    print("- Impossible quantity in transactions (-4): 5 rows")
    print("- Inventory arithmetic mismatches: 20 rows")

if __name__ == "__main__":
    main()
