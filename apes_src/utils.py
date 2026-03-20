from pyhocon import ConfigFactory

def parse_configs(path):
	f = open(path)
	conf_text = f.read()
	f.close()

	conf = ConfigFactory.parse_string(conf_text)
	return conf, conf_text

def save_config(args, exp_name, conf_text):
    conf_name = args.conf_path.split("/")[-1]
    save_path = f"exp/{exp_name}"
    os.makedirs(save_path, exist_ok=True)
    
    save_path = os.path.join(save_path, conf_name)
    write_file(save_path, conf_text)