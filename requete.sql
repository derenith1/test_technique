/*create table transactions(date date, order_id integer, client_id integer, prod_id integer, prod_price float, prod_qty integer);
create table product_nomenclature(product_id integer, product_type varchar(10), product_name varchar(100));

insert into transactions(date, order_id, client_id, prod_id, prod_price, prod_qty) values('2019-01-01', 1234, 999, 490756, 50, 1);
insert into transactions(date, order_id, client_id, prod_id, prod_price, prod_qty) values('2019-01-01', 1234, 999, 389728, 3.56, 4);
insert into transactions(date, order_id, client_id, prod_id, prod_price, prod_qty) values('2019-02-01', 3456, 845, 490756, 50, 2);
insert into transactions(date, order_id, client_id, prod_id, prod_price, prod_qty) values('2019-02-01', 3456, 845, 549380, 300, 1);
insert into transactions(date, order_id, client_id, prod_id, prod_price, prod_qty) values('2019-03-01', 3456, 845, 293718, 10, 6);
insert into transactions(date, order_id, client_id, prod_id, prod_price, prod_qty) values('2019-12-31', 789, 999, 125578, 15, 1);
insert into transactions(date, order_id, client_id, prod_id, prod_price, prod_qty) values('2020-12-01', 789, 999, 787878, 75, 1);

insert into product_nomenclature(product_id, product_type, product_name) values(490756, 'MEUBLE', 'Chaise');
insert into product_nomenclature(product_id, product_type, product_name) values(389728, 'DECO', 'Boule de Noel');
insert into product_nomenclature(product_id, product_type, product_name) values(549380, 'MEUBLE', 'Canapé');
insert into product_nomenclature(product_id, product_type, product_name) values(293718, 'DECO', 'Mug');
insert into product_nomenclature(product_id, product_type, product_name) values(125578, 'MLP', 'Wok');
insert into product_nomenclature(product_id, product_type, product_name) values(787878, 'MEUBLE', 'Table');
*/

-- First query
select date, sum(prod_price*prod_qty) as CA 
from transactions
where date between '2019-01-01' and '2019-12-31'
group by date;


-- Second query
select table_meuble.client_id, CAM, CAD from (
    select client_id, SUM(prod_qty) as CAM from transactions t
    join product_nomenclature p
    where p.product_id = t.prod_id 
    and product_type = 'MEUBLE'
    and date between '2019-01-01' and '2019-12-31'
    group by client_id
) as table_meuble
join (
    select client_id, SUM(prod_qty) as CAD from transactions t
    join product_nomenclature p
    where p.product_id = t.prod_id 
    and product_type = 'DECO'
    and date between '2019-01-01' and '2019-12-31'
    group by client_id
) as table_deco
where table_meuble.client_id = table_deco.client_id
group by client_id;
