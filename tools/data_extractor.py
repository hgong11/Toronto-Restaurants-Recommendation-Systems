import json
import os
root_dir = os.path.dirname(os.path.dirname(__file__))


def main():
    print('Loading books from files.json...')

    # Extract from business list
    business_id_list = []
    business_list = []
    business_counter = 0

    with open('{}/yelp_dataset/yelp_academic_dataset_business.json'.format(root_dir), 'r', encoding='utf-8') as business_file:
        for data in business_file:
            business_counter += 1
            json_data = json.loads(data)
            if json_data["city"] == "Toronto" and str(json_data["categories"]).find("Restaurants") != -1 and json_data["is_open"] == 1:
                business_id_list.append(json_data["business_id"])
                business_list.append(json.dumps(json_data, ensure_ascii=False))

    print("Total business: " + str(business_counter))
    print("Selected business: " + str(len(business_id_list)))

    f = open("{}/tools/business.json".format(root_dir), "a")
    for business in business_list:
        f.write(f"{business}\n")

    f.close()

    # Extract from review list
    review_list = []
    userId_set = set()
    review_counter = 0
    with open('{}/yelp_dataset/yelp_academic_dataset_review.json'.format(root_dir), 'r', encoding='utf-8') as review_file:
        for data in review_file:
            review_counter += 1
            json_data = json.loads(data)
            if json_data["business_id"] in business_id_list:
                review_list.append(json.dumps(json_data, ensure_ascii=False))
                userId_set.add(json_data["user_id"])

    print("Total reviews: " + str(review_counter))
    print("Selected reviews: " + str(len(review_list)))

    f = open("{}/tools/reviews.json".format(root_dir), "a")
    for review in review_list:
        f.write(f"{review}\n")

    f.close()

    # Extract from tip list
    tips_list = []
    tip_counter = 0
    with open('{}/yelp_dataset/yelp_academic_dataset_tip.json'.format(root_dir), 'r', encoding='utf-8') as tips_file:
        for data in tips_file:
            tip_counter += 1
            json_data = json.loads(data)
            if json_data["business_id"] in business_id_list:
                tips_list.append(json.dumps(json_data, ensure_ascii=False))

    print("Total tips: " + str(tip_counter))
    print("Selected tips: " + str(len(tips_list)))

    f = open("{}/tools/tips.json".format(root_dir), "a")
    for tip in tips_list:
        f.write(f"{tip}\n")

    f.close()

    # Extract from user list
    user_list = []
    user_counter = 0
    with open('{}/yelp_dataset/yelp_academic_dataset_user.json'.format(root_dir), 'r', encoding='utf-8') as user_file:
        for data in user_file:
            user_counter += 1
            json_data = json.loads(data)
            if json_data["user_id"] in userId_set:
                user_list.append(json.dumps(json_data, ensure_ascii=False))

    print("Total user: " + str(user_counter))
    print("Selected users: " + str(len(user_list)))

    f = open("{}/tools/users.json".format(root_dir), "a")
    for user in user_list:
        f.write(f"{user}\n")

    f.close()


if __name__ == "__main__":
    main()