"""
Bagisto Test Data Population Script
Populates database with realistic test data for chatbot testing
"""

import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta
import random


class BagistoDataPopulator:
    """Populates Bagisto database with test data"""

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

    def _execute(self, query, params=None, commit=True):
        """Execute SQL query with optional commit"""
        try:
            self.cursor.execute(query, params or ())
            if commit:
                self.connection.commit()
            return True
        except Error as e:
            print(f"Execute error: {e}")
            print(f"Query: {query[:200]}...")
            self.connection.rollback()
            return False

    def _execute_query(self, query, params=None):
        """Execute SELECT query"""
        try:
            self.cursor.execute(query, params or ())
            return self.cursor.fetchall()
        except Error as e:
            print(f"Query error: {e}")
            return None

    def get_existing_products(self):
        """Get existing products from database"""
        query = """
            SELECT
                pf.product_id as id,
                pf.sku,
                pf.name,
                pf.price
            FROM product_flat pf
            WHERE pf.status = 1
              AND pf.locale = 'en'
        """
        return self._execute_query(query) or []

    def get_existing_customers(self):
        """Get existing customers"""
        query = "SELECT id, first_name, last_name, email FROM customers"
        return self._execute_query(query) or []

    def get_channel_id(self):
        """Get default channel ID"""
        query = "SELECT id FROM channels LIMIT 1"
        result = self._execute_query(query)
        return result[0]['id'] if result else 1

    def get_inventory_source_id(self):
        """Get default inventory source ID"""
        query = "SELECT id FROM inventory_sources WHERE status = 1 LIMIT 1"
        result = self._execute_query(query)
        return result[0]['id'] if result else 1

    def get_next_increment_id(self):
        """Get next order increment ID"""
        query = "SELECT MAX(CAST(increment_id AS UNSIGNED)) as max_id FROM orders"
        result = self._execute_query(query)
        max_id = result[0]['max_id'] if result and result[0]['max_id'] else 0
        return max_id + 1

    def update_product_inventories(self):
        """Ensure all products have stock"""
        print("\n1. UPDATING PRODUCT INVENTORIES")
        print("-" * 40)

        products = self.get_existing_products()
        source_id = self.get_inventory_source_id()

        stock_levels = [10, 15, 20, 25, 30, 50, 100, 5, 8, 12, 0]  # Include some low/zero stock

        for i, product in enumerate(products):
            stock = stock_levels[i % len(stock_levels)]

            # Check if inventory record exists
            query = "SELECT id FROM product_inventories WHERE product_id = %s AND inventory_source_id = %s"
            result = self._execute_query(query, (product['id'], source_id))

            if result:
                # Update existing
                update_query = """
                    UPDATE product_inventories
                    SET qty = %s
                    WHERE product_id = %s AND inventory_source_id = %s
                """
                self._execute(update_query, (stock, product['id'], source_id))
            else:
                # Insert new
                insert_query = """
                    INSERT INTO product_inventories (product_id, inventory_source_id, qty)
                    VALUES (%s, %s, %s)
                """
                self._execute(insert_query, (product['id'], source_id, stock))

            print(f"  {product['name'][:40]}: {stock} units")

        print(f"\nUpdated stock for {len(products)} products")

    def update_product_prices(self):
        """Update product prices to realistic Indonesian Rupiah values"""
        print("\n2. UPDATING PRODUCT PRICES")
        print("-" * 40)

        # Map product types to realistic IDR prices
        price_map = {
            'beanie': 89000,
            'jacket': 349000,
            'hoodie': 299000,
            'sweater': 249000,
            'coat': 499000,
            'parka': 599000,
            'scarf': 79000,
            'gloves': 69000,
            'cardigan': 279000,
        }

        products = self.get_existing_products()

        for product in products:
            name_lower = product['name'].lower()
            new_price = None

            for keyword, price in price_map.items():
                if keyword in name_lower:
                    new_price = price
                    break

            if new_price is None:
                # Default pricing based on random range
                new_price = random.choice([99000, 149000, 199000, 249000, 299000])

            # Update price
            update_query = """
                UPDATE product_flat
                SET price = %s
                WHERE product_id = %s
            """
            self._execute(update_query, (new_price, product['id']))
            print(f"  {product['name'][:40]}: IDR {new_price:,}")

        print(f"\nUpdated prices for {len(products)} products")

    def add_customers(self):
        """Add Indonesian customers for testing"""
        print("\n3. ADDING CUSTOMERS")
        print("-" * 40)

        customers_to_add = [
            {
                'first_name': 'Budi',
                'last_name': 'Santoso',
                'email': 'budi.santoso@email.com',
                'phone': '081234567890'
            },
            {
                'first_name': 'Siti',
                'last_name': 'Rahayu',
                'email': 'siti.rahayu@email.com',
                'phone': '082345678901'
            },
            {
                'first_name': 'Ahmad',
                'last_name': 'Wijaya',
                'email': 'ahmad.wijaya@email.com',
                'phone': '083456789012'
            },
            {
                'first_name': 'Dewi',
                'last_name': 'Kusuma',
                'email': 'dewi.kusuma@email.com',
                'phone': '084567890123'
            },
            {
                'first_name': 'Rudi',
                'last_name': 'Hermawan',
                'email': 'rudi.hermawan@email.com',
                'phone': '085678901234'
            },
        ]

        channel_id = self.get_channel_id()
        added_count = 0

        for customer in customers_to_add:
            # Check if email already exists
            query = "SELECT id FROM customers WHERE email = %s"
            result = self._execute_query(query, (customer['email'],))

            if result:
                print(f"  {customer['first_name']} {customer['last_name']}: Already exists")
                continue

            # Insert customer
            insert_query = """
                INSERT INTO customers (
                    first_name, last_name, email, phone, gender,
                    is_verified, channel_id, status, created_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, 'Male',
                    1, %s, 1, NOW(), NOW()
                )
            """
            if self._execute(insert_query, (
                customer['first_name'],
                customer['last_name'],
                customer['email'],
                customer['phone'],
                channel_id
            )):
                print(f"  Added: {customer['first_name']} {customer['last_name']}")
                added_count += 1

        print(f"\nAdded {added_count} new customers")

    def create_orders(self):
        """Create orders with various statuses"""
        print("\n4. CREATING ORDERS")
        print("-" * 40)

        customers = self.get_existing_customers()
        products = self.get_existing_products()
        channel_id = self.get_channel_id()

        if not customers or not products:
            print("ERROR: No customers or products found!")
            return

        # Order configurations with different statuses
        order_configs = [
            {'status': 'pending', 'qty': 1, 'days_ago': 1},
            {'status': 'pending', 'qty': 2, 'days_ago': 2},
            {'status': 'pending', 'qty': 1, 'days_ago': 3},
            {'status': 'processing', 'qty': 1, 'days_ago': 4},
            {'status': 'processing', 'qty': 3, 'days_ago': 5},
            {'status': 'processing', 'qty': 1, 'days_ago': 6},
            {'status': 'completed', 'qty': 1, 'days_ago': 10},
            {'status': 'completed', 'qty': 2, 'days_ago': 14},
            {'status': 'canceled', 'qty': 1, 'days_ago': 7},
            {'status': 'pending', 'qty': 4, 'days_ago': 0},  # Multi-item order
        ]

        created_count = 0

        for i, config in enumerate(order_configs):
            customer = customers[i % len(customers)]
            product = products[i % len(products)]

            # Calculate order date
            order_date = datetime.now() - timedelta(days=config['days_ago'])

            # Get next increment ID
            increment_id = self.get_next_increment_id()

            # Calculate totals
            price = float(product['price']) if product['price'] else 149000
            qty = config['qty']
            subtotal = price * qty
            shipping = 15000 if subtotal < 200000 else 0  # Free shipping over 200k
            grand_total = subtotal + shipping

            # Insert order
            order_query = """
                INSERT INTO orders (
                    increment_id,
                    status,
                    channel_id,
                    customer_id,
                    customer_email,
                    customer_first_name,
                    customer_last_name,
                    is_guest,
                    shipping_method,
                    shipping_title,
                    shipping_description,
                    shipping_amount,
                    base_shipping_amount,
                    total_item_count,
                    total_qty_ordered,
                    base_currency_code,
                    channel_currency_code,
                    order_currency_code,
                    grand_total,
                    base_grand_total,
                    sub_total,
                    base_sub_total,
                    tax_amount,
                    base_tax_amount,
                    discount_amount,
                    base_discount_amount,
                    created_at,
                    updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    0, 'flatrate_flatrate', 'Flat Rate - Flat Rate', 'Flat Rate Shipping', %s, %s,
                    %s, %s, 'IDR', 'IDR', 'IDR',
                    %s, %s, %s, %s, 0, 0, 0, 0,
                    %s, %s
                )
            """

            if not self._execute(order_query, (
                str(increment_id),
                config['status'],
                channel_id,
                customer['id'],
                customer['email'],
                customer['first_name'],
                customer['last_name'],
                shipping,
                shipping,
                1,  # total_item_count (will be items in order_items)
                qty,  # total_qty_ordered
                grand_total,
                grand_total,
                subtotal,
                subtotal,
                order_date,
                order_date
            )):
                continue

            # Get the order ID
            order_id = self.cursor.lastrowid

            # Insert order items
            for item_idx in range(config['qty'] if config['qty'] <= 3 else 1):
                item_product = products[(i + item_idx) % len(products)]
                item_price = float(item_product['price']) if item_product['price'] else 149000
                item_qty = 1 if config['qty'] <= 3 else config['qty']

                item_query = """
                    INSERT INTO order_items (
                        order_id,
                        product_id,
                        sku,
                        type,
                        name,
                        qty_ordered,
                        qty_shipped,
                        qty_invoiced,
                        qty_canceled,
                        qty_refunded,
                        price,
                        base_price,
                        total,
                        base_total,
                        tax_amount,
                        base_tax_amount,
                        discount_amount,
                        base_discount_amount,
                        created_at,
                        updated_at
                    ) VALUES (
                        %s, %s, %s, 'simple', %s,
                        %s, 0, 0, 0, 0,
                        %s, %s, %s, %s,
                        0, 0, 0, 0,
                        %s, %s
                    )
                """
                self._execute(item_query, (
                    order_id,
                    item_product['id'],
                    item_product['sku'],
                    item_product['name'],
                    item_qty,
                    item_price,
                    item_price,
                    item_price * item_qty,
                    item_price * item_qty,
                    order_date,
                    order_date
                ))

            # Insert order payment
            payment_method = random.choice(['cashondelivery', 'moneytransfer', 'moneytransfer'])
            payment_query = """
                INSERT INTO order_payment (
                    order_id, method, method_title, created_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s
                )
            """
            method_title = 'Cash on Delivery' if payment_method == 'cashondelivery' else 'Bank Transfer'
            self._execute(payment_query, (order_id, payment_method, method_title, order_date, order_date))

            print(f"  Order #{increment_id}: {config['status']} - {customer['first_name']} - IDR {grand_total:,.0f}")
            created_count += 1

        print(f"\nCreated {created_count} orders")

    def verify_data(self):
        """Verify the populated data"""
        print("\n5. VERIFICATION")
        print("-" * 40)

        # Count orders by status
        query = "SELECT status, COUNT(*) as cnt FROM orders GROUP BY status"
        result = self._execute_query(query)
        print("\nOrders by status:")
        for row in result or []:
            print(f"  {row['status']}: {row['cnt']}")

        # Count products with stock
        query = """
            SELECT COUNT(DISTINCT product_id) as cnt
            FROM product_inventories
            WHERE qty > 0
        """
        result = self._execute_query(query)
        stock_count = result[0]['cnt'] if result else 0
        print(f"\nProducts with stock: {stock_count}")

        # Total customers
        query = "SELECT COUNT(*) as cnt FROM customers"
        result = self._execute_query(query)
        customer_count = result[0]['cnt'] if result else 0
        print(f"Total customers: {customer_count}")

        # Total orders
        query = "SELECT COUNT(*) as cnt FROM orders"
        result = self._execute_query(query)
        order_count = result[0]['cnt'] if result else 0
        print(f"Total orders: {order_count}")

        # Sample order details
        print("\nSample orders:")
        query = """
            SELECT o.id, o.increment_id, o.status, o.grand_total,
                   o.customer_first_name, o.customer_last_name
            FROM orders o
            ORDER BY o.created_at DESC
            LIMIT 5
        """
        result = self._execute_query(query)
        for row in result or []:
            print(f"  #{row['increment_id']}: {row['status']} - {row['customer_first_name']} - IDR {float(row['grand_total']):,.0f}")

    def run_all(self):
        """Run all population steps"""
        print("="*70)
        print("BAGISTO TEST DATA POPULATION")
        print("="*70)

        self.update_product_inventories()
        self.update_product_prices()
        self.add_customers()
        self.create_orders()
        self.verify_data()

        print("\n" + "="*70)
        print("DATA POPULATION COMPLETE")
        print("="*70)

    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("\nConnection closed")


if __name__ == "__main__":
    populator = BagistoDataPopulator()
    populator.run_all()
    populator.close()
