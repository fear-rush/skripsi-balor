"""
Bagisto Database Schema Inspector
Analyzes tables, columns, relationships and constraints
"""

import mysql.connector
from mysql.connector import Error
from typing import Dict, List, Any
import json


class BagistoSchemaInspector:
    """Inspects Bagisto database schema for test data population planning"""

    def __init__(self, db_config=None):
        if db_config is None:
            db_config = {
                'host': 'localhost',
                'user': 'root',
                'password': '',
                'database': 'bagisto_db',
                'port': 3306
            }

        self.db_config = db_config
        self.connection = None
        self.cursor = None
        self._connect()

    def _connect(self):
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            if self.connection.is_connected():
                self.cursor = self.connection.cursor(dictionary=True)
                print(f"Connected to {self.db_config['database']}")
                return True
        except Error as e:
            print(f"Connection error: {e}")
            return False

    def _execute_query(self, query, params=None):
        """Execute SQL query"""
        try:
            self.cursor.execute(query, params or ())
            return self.cursor.fetchall()
        except Error as e:
            print(f"Query error: {e}")
            return None

    def get_all_tables(self) -> List[str]:
        """Get all table names"""
        query = "SHOW TABLES"
        result = self._execute_query(query)
        if result:
            return [list(row.values())[0] for row in result]
        return []

    def get_table_columns(self, table_name: str) -> List[Dict]:
        """Get columns info for a table"""
        query = f"DESCRIBE {table_name}"
        return self._execute_query(query) or []

    def get_table_row_count(self, table_name: str) -> int:
        """Get row count for a table"""
        query = f"SELECT COUNT(*) as cnt FROM {table_name}"
        result = self._execute_query(query)
        return result[0]['cnt'] if result else 0

    def get_foreign_keys(self, table_name: str) -> List[Dict]:
        """Get foreign key relationships for a table"""
        query = """
            SELECT
                COLUMN_NAME,
                REFERENCED_TABLE_NAME,
                REFERENCED_COLUMN_NAME
            FROM information_schema.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = %s
              AND TABLE_NAME = %s
              AND REFERENCED_TABLE_NAME IS NOT NULL
        """
        return self._execute_query(query, (self.db_config['database'], table_name)) or []

    def get_sample_data(self, table_name: str, limit: int = 3) -> List[Dict]:
        """Get sample data from a table"""
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        return self._execute_query(query) or []

    def inspect_relevant_tables(self) -> Dict[str, Any]:
        """Inspect tables relevant to chatbot testing"""
        relevant_tables = [
            'orders',
            'order_items',
            'order_payment',
            'products',
            'product_flat',
            'product_inventories',
            'customers',
            'addresses',
            'core_config',
            'channels',
            'categories',
            'product_categories'
        ]

        schema_info = {}

        for table in relevant_tables:
            print(f"\nInspecting: {table}")

            # Check if table exists
            all_tables = self.get_all_tables()
            if table not in all_tables:
                print(f"  Table {table} not found, skipping...")
                continue

            columns = self.get_table_columns(table)
            row_count = self.get_table_row_count(table)
            foreign_keys = self.get_foreign_keys(table)
            sample = self.get_sample_data(table, 2)

            schema_info[table] = {
                'row_count': row_count,
                'columns': [
                    {
                        'name': col['Field'],
                        'type': col['Type'],
                        'null': col['Null'],
                        'key': col['Key'],
                        'default': col['Default'],
                        'extra': col['Extra']
                    }
                    for col in columns
                ],
                'foreign_keys': foreign_keys,
                'sample_data': sample
            }

            print(f"  Rows: {row_count}")
            print(f"  Columns: {len(columns)}")
            print(f"  Foreign Keys: {len(foreign_keys)}")

        return schema_info

    def get_order_statuses(self) -> List[str]:
        """Get unique order statuses in the database"""
        query = "SELECT DISTINCT status FROM orders"
        result = self._execute_query(query)
        return [row['status'] for row in result] if result else []

    def get_product_categories(self) -> List[Dict]:
        """Get product categories"""
        query = """
            SELECT
                c.id,
                ct.name,
                ct.slug,
                COUNT(pc.product_id) as product_count
            FROM categories c
            LEFT JOIN category_translations ct ON c.id = ct.category_id
            LEFT JOIN product_categories pc ON c.id = pc.category_id
            WHERE ct.locale = 'en'
            GROUP BY c.id, ct.name, ct.slug
        """
        return self._execute_query(query) or []

    def get_inventory_sources(self) -> List[Dict]:
        """Get inventory sources"""
        query = "SELECT * FROM inventory_sources WHERE status = 1"
        return self._execute_query(query) or []

    def get_channels(self) -> List[Dict]:
        """Get available channels"""
        query = "SELECT id, code, name, root_category_id FROM channels"
        return self._execute_query(query) or []

    def analyze_for_test_data(self) -> Dict[str, Any]:
        """Comprehensive analysis for test data population"""
        print("\n" + "="*70)
        print("BAGISTO DATABASE ANALYSIS FOR TEST DATA POPULATION")
        print("="*70)

        analysis = {}

        # 1. Current data counts
        print("\n1. CURRENT DATA COUNTS")
        print("-" * 40)
        count_tables = ['orders', 'order_items', 'products', 'product_flat',
                       'product_inventories', 'customers', 'addresses']

        analysis['current_counts'] = {}
        for table in count_tables:
            try:
                count = self.get_table_row_count(table)
                analysis['current_counts'][table] = count
                print(f"  {table}: {count}")
            except:
                print(f"  {table}: ERROR")

        # 2. Order statuses
        print("\n2. ORDER STATUSES IN USE")
        print("-" * 40)
        statuses = self.get_order_statuses()
        analysis['order_statuses'] = statuses
        for status in statuses:
            print(f"  - {status}")

        # 3. Categories
        print("\n3. PRODUCT CATEGORIES")
        print("-" * 40)
        categories = self.get_product_categories()
        analysis['categories'] = categories
        for cat in categories:
            print(f"  - {cat.get('name', 'Unknown')} ({cat.get('product_count', 0)} products)")

        # 4. Inventory sources
        print("\n4. INVENTORY SOURCES")
        print("-" * 40)
        sources = self.get_inventory_sources()
        analysis['inventory_sources'] = sources
        for src in sources:
            print(f"  - {src.get('name', 'Unknown')} (ID: {src.get('id')})")

        # 5. Channels
        print("\n5. SALES CHANNELS")
        print("-" * 40)
        channels = self.get_channels()
        analysis['channels'] = channels
        for ch in channels:
            print(f"  - {ch.get('name', 'Unknown')} (code: {ch.get('code')})")

        # 6. Required fields analysis
        print("\n6. REQUIRED FIELDS FOR DATA INSERTION")
        print("-" * 40)

        # Inspect key tables
        key_tables = ['orders', 'order_items', 'customers', 'product_flat', 'product_inventories']
        analysis['required_fields'] = {}

        for table in key_tables:
            try:
                columns = self.get_table_columns(table)
                required = [col['Field'] for col in columns
                           if col['Null'] == 'NO' and col['Default'] is None
                           and 'auto_increment' not in (col['Extra'] or '')]
                analysis['required_fields'][table] = required
                print(f"\n  {table}:")
                if required:
                    for field in required[:10]:  # Limit to 10
                        print(f"    - {field}")
                    if len(required) > 10:
                        print(f"    ... and {len(required) - 10} more")
                else:
                    print("    (no strictly required fields)")
            except:
                pass

        # 7. Sample existing data
        print("\n7. SAMPLE EXISTING DATA")
        print("-" * 40)

        # Sample order
        orders = self.get_sample_data('orders', 1)
        if orders:
            order = orders[0]
            print(f"\n  Sample Order:")
            print(f"    ID: {order.get('id')}")
            print(f"    Status: {order.get('status')}")
            print(f"    Customer: {order.get('customer_first_name')} {order.get('customer_last_name')}")
            print(f"    Total: {order.get('grand_total')}")

        # Sample product
        products = self.get_sample_data('product_flat', 1)
        if products:
            product = products[0]
            print(f"\n  Sample Product:")
            print(f"    ID: {product.get('product_id')}")
            print(f"    Name: {product.get('name')}")
            print(f"    SKU: {product.get('sku')}")
            print(f"    Price: {product.get('price')}")

        # Sample customer
        customers = self.get_sample_data('customers', 1)
        if customers:
            customer = customers[0]
            print(f"\n  Sample Customer:")
            print(f"    ID: {customer.get('id')}")
            print(f"    Name: {customer.get('first_name')} {customer.get('last_name')}")
            print(f"    Email: {customer.get('email')}")

        return analysis

    def generate_schema_report(self, output_file: str = None):
        """Generate comprehensive schema report"""
        schema = self.inspect_relevant_tables()
        analysis = self.analyze_for_test_data()

        report = {
            'schema': schema,
            'analysis': analysis
        }

        if output_file:
            # Convert datetime objects to strings for JSON serialization
            def serialize(obj):
                if hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                elif isinstance(obj, bytes):
                    return obj.decode('utf-8', errors='replace')
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=serialize)
            print(f"\nReport saved to: {output_file}")

        return report

    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("Connection closed")


if __name__ == "__main__":
    print("="*70)
    print("BAGISTO DATABASE SCHEMA INSPECTOR")
    print("="*70)

    inspector = BagistoSchemaInspector()

    # Run comprehensive analysis
    report = inspector.generate_schema_report(
        output_file='evaluation/bagisto_schema_report.json'
    )

    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR TEST DATA")
    print("="*70)
    print("""
Based on the analysis:

1. ORDERS - Need diverse statuses:
   - 3 pending orders (waiting for payment)
   - 3 processing orders (being prepared)
   - 2 completed orders (delivered)
   - 1 cancelled order
   - 1 order with multiple items

2. PRODUCTS - Need varied catalog:
   - Use existing products where possible
   - Ensure varied prices (cheap to expensive)
   - Update stock levels in product_inventories

3. CUSTOMERS - Need realistic data:
   - Indonesian names
   - Valid email formats
   - Complete addresses

4. KEY RELATIONSHIPS:
   - orders.customer_id -> customers.id
   - order_items.order_id -> orders.id
   - order_items.product_id -> products.id
   - product_inventories.product_id -> products.id
""")

    inspector.close()
