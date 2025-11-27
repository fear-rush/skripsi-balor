"""
Bagisto Database Inspector
Generates detailed report of database structure
"""

import mysql.connector
from mysql.connector import Error
import json

def inspect_bagisto_database():
    """Inspect Bagisto database and generate report"""
    
    # Database config
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': '',
        'database': 'bagisto_db',
        'port': 3306
    }
    
    try:
        # Connect
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)
        
        print("="*70)
        print("📊 BAGISTO DATABASE INSPECTION REPORT")
        print("="*70)
        
        report = {}
        
        # ========================================================
        # 1. LIST ALL TABLES
        # ========================================================
        
        print("\n1️⃣  ALL TABLES IN DATABASE")
        print("-"*70)
        
        cursor.execute("SHOW TABLES")
        tables = [list(row.values())[0] for row in cursor.fetchall()]
        
        print(f"Total tables: {len(tables)}\n")
        
        # Group tables by category
        order_tables = [t for t in tables if 'order' in t]
        product_tables = [t for t in tables if 'product' in t]
        customer_tables = [t for t in tables if 'customer' in t or 'user' in t]
        payment_tables = [t for t in tables if 'payment' in t or 'config' in t]
        
        print("📦 Order-related tables:")
        for table in order_tables:
            print(f"   - {table}")
        
        print("\n🛍️  Product-related tables:")
        for table in product_tables:
            print(f"   - {table}")
        
        print("\n👤 Customer-related tables:")
        for table in customer_tables:
            print(f"   - {table}")
        
        print("\n💳 Payment/Config tables:")
        for table in payment_tables:
            print(f"   - {table}")
        
        report['tables'] = {
            'all': tables,
            'orders': order_tables,
            'products': product_tables,
            'customers': customer_tables,
            'payments': payment_tables
        }
        
        # ========================================================
        # 2. ORDERS TABLE STRUCTURE
        # ========================================================
        
        print("\n\n2️⃣  ORDERS TABLE STRUCTURE")
        print("-"*70)
        
        if 'orders' in tables:
            cursor.execute("DESCRIBE orders")
            orders_columns = cursor.fetchall()
            
            print("Columns:")
            for col in orders_columns:
                print(f"   • {col['Field']:<30} {col['Type']:<20} {'NULL' if col['Null'] == 'YES' else 'NOT NULL':<10} {col['Key']}")
            
            # Sample data
            cursor.execute("SELECT * FROM orders LIMIT 3")
            orders_sample = cursor.fetchall()
            
            print(f"\nSample data ({len(orders_sample)} rows):")
            if orders_sample:
                for order in orders_sample:
                    print(f"\n   Order ID: {order.get('id')}")
                    print(f"   Status: {order.get('status')}")
                    print(f"   Customer: {order.get('customer_first_name')} {order.get('customer_last_name')}")
                    print(f"   Total: {order.get('grand_total')}")
                    print(f"   Date: {order.get('created_at')}")
            else:
                print("   ⚠️  No sample data available")
            
            report['orders'] = {
                'columns': orders_columns,
                'sample': orders_sample
            }
        
        # ========================================================
        # 3. PRODUCTS TABLE STRUCTURE
        # ========================================================
        
        print("\n\n3️⃣  PRODUCTS TABLE STRUCTURE")
        print("-"*70)
        
        if 'products' in tables:
            cursor.execute("DESCRIBE products")
            products_columns = cursor.fetchall()
            
            print("Products table columns:")
            for col in products_columns:
                print(f"   • {col['Field']:<30} {col['Type']:<20} {col['Key']}")
        
        if 'product_flat' in tables:
            cursor.execute("DESCRIBE product_flat")
            product_flat_columns = cursor.fetchall()
            
            print("\nProduct_flat table columns:")
            for col in product_flat_columns:
                print(f"   • {col['Field']:<30} {col['Type']:<20} {col['Key']}")
            
            # Sample data
            cursor.execute("""
                SELECT 
                    pf.id,
                    pf.product_id,
                    pf.sku,
                    pf.name,
                    pf.price,
                    pf.short_description
                FROM product_flat pf
                LIMIT 3
            """)
            products_sample = cursor.fetchall()
            
            print(f"\nSample product data ({len(products_sample)} rows):")
            for product in products_sample:
                print(f"\n   ID: {product.get('product_id')} | SKU: {product.get('sku')}")
                print(f"   Name: {product.get('name')}")
                print(f"   Price: {product.get('price')}")
            
            report['products'] = {
                'columns': product_flat_columns,
                'sample': products_sample
            }
        
        # ========================================================
        # 4. PRODUCT INVENTORY
        # ========================================================
        
        print("\n\n4️⃣  PRODUCT INVENTORY")
        print("-"*70)
        
        if 'product_inventories' in tables:
            cursor.execute("DESCRIBE product_inventories")
            inventory_columns = cursor.fetchall()
            
            print("Inventory columns:")
            for col in inventory_columns:
                print(f"   • {col['Field']:<30} {col['Type']:<20}")
            
            # Sample data
            cursor.execute("""
                SELECT 
                    pi.product_id,
                    pi.qty,
                    p.sku,
                    pf.name
                FROM product_inventories pi
                LEFT JOIN products p ON pi.product_id = p.id
                LEFT JOIN product_flat pf ON p.id = pf.product_id
                LIMIT 5
            """)
            inventory_sample = cursor.fetchall()
            
            print(f"\nSample inventory data:")
            for inv in inventory_sample:
                print(f"   Product ID {inv.get('product_id')}: {inv.get('name')} - Stock: {inv.get('qty')}")
            
            report['inventory'] = {
                'columns': inventory_columns,
                'sample': inventory_sample
            }
        
        # ========================================================
        # 5. PAYMENT CONFIGURATION
        # ========================================================
        
        print("\n\n5️⃣  PAYMENT CONFIGURATION")
        print("-"*70)
        
        if 'core_config' in tables:
            cursor.execute("""
                SELECT code, value
                FROM core_config
                WHERE code LIKE '%payment%'
                   OR code LIKE '%sales.paymentmethods%'
                LIMIT 20
            """)
            payment_config = cursor.fetchall()
            
            print("Payment-related configurations:")
            for config in payment_config:
                print(f"   • {config['code']}: {config['value']}")
            
            report['payment_config'] = payment_config
        
        # ========================================================
        # 6. ORDER STATUSES
        # ========================================================
        
        print("\n\n6️⃣  ORDER STATUSES (DISTINCT VALUES)")
        print("-"*70)
        
        if 'orders' in tables:
            cursor.execute("SELECT DISTINCT status FROM orders")
            statuses = [row['status'] for row in cursor.fetchall()]
            
            print("Available order statuses:")
            for status in statuses:
                print(f"   • {status}")
            
            report['order_statuses'] = statuses
        
        # ========================================================
        # 7. SAVE REPORT TO FILE
        # ========================================================
        
        print("\n\n7️⃣  SAVING REPORT")
        print("-"*70)
        
        # Save as JSON
        with open('bagisto_database_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("✅ Report saved to: bagisto_database_report.json")
        
        # Save as readable text
        with open('bagisto_database_report.txt', 'w') as f:
            f.write("="*70 + "\n")
            f.write("BAGISTO DATABASE STRUCTURE REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("ALL TABLES:\n")
            for table in tables:
                f.write(f"  - {table}\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print("✅ Readable report saved to: bagisto_database_report.txt")
        
        # Close connection
        cursor.close()
        connection.close()
        
        print("\n" + "="*70)
        print("✅ INSPECTION COMPLETE!")
        print("="*70)
        print("\n📧 Share the following files with me:")
        print("   1. bagisto_database_report.json")
        print("   2. bagisto_database_report.txt")
        print("\n   Or copy-paste the output above!")
        
    except Error as e:
        print(f"❌ Database error: {e}")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    inspect_bagisto_database()