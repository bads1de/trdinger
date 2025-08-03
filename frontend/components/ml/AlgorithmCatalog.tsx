import React, { useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Chip,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Alert,
  CircularProgress,
  Paper,
  Divider,
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  TrendingUp as TrendingUpIcon,
  Speed as SpeedIcon,
  Accuracy as AccuracyIcon,
  Memory as MemoryIcon,
  Psychology as PsychologyIcon,
} from '@mui/icons-material';
import { useAlgorithms, Algorithm } from '../../hooks/useAlgorithms';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`algorithm-tabpanel-${index}`}
      aria-labelledby={`algorithm-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const AlgorithmCatalog: React.FC = () => {
  const {
    algorithms,
    algorithmsByType,
    statistics,
    searchAlgorithms,
    isLoading,
    error,
    getTypeLabel,
  } = useAlgorithms();

  const [searchQuery, setSearchQuery] = useState('');
  const [selectedType, setSelectedType] = useState<string>('all');
  const [tabValue, setTabValue] = useState(0);

  // フィルタリングされたアルゴリズム
  const filteredAlgorithms = React.useMemo(() => {
    let filtered = searchAlgorithms(searchQuery);

    if (selectedType !== 'all') {
      filtered = filtered.filter(algo => algo.type === selectedType);
    }

    return filtered;
  }, [searchAlgorithms, searchQuery, selectedType]);

  // タイプアイコンの取得
  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'tree_based':
        return <TrendingUpIcon />;
      case 'linear':
        return <SpeedIcon />;
      case 'boosting':
        return <AccuracyIcon />;
      case 'probabilistic':
        return <PsychologyIcon />;
      case 'instance_based':
        return <MemoryIcon />;
      default:
        return <InfoIcon />;
    }
  };

  // アルゴリズムカードコンポーネント
  const AlgorithmCard: React.FC<{ algorithm: Algorithm }> = ({ algorithm }) => (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardContent sx={{ flexGrow: 1 }}>
        <Box display="flex" alignItems="center" mb={2}>
          {getTypeIcon(algorithm.type)}
          <Typography variant="h6" component="h3" sx={{ ml: 1 }}>
            {algorithm.display_name}
          </Typography>
        </Box>

        <Typography variant="body2" color="text.secondary" mb={2}>
          {algorithm.description}
        </Typography>

        <Box mb={2}>
          <Chip
            label={getTypeLabel(algorithm.type)}
            size="small"
            variant="outlined"
            sx={{ mr: 1, mb: 1 }}
          />
          {algorithm.has_probability_prediction && (
            <Chip
              label="確率予測"
              size="small"
              color="info"
              sx={{ mr: 1, mb: 1 }}
            />
          )}
          {algorithm.has_feature_importance && (
            <Chip
              label="特徴量重要度"
              size="small"
              color="success"
              sx={{ mr: 1, mb: 1 }}
            />
          )}
          {algorithm.note && (
            <Chip
              label="注意事項あり"
              size="small"
              color="warning"
              sx={{ mr: 1, mb: 1 }}
            />
          )}
        </Box>

        <Typography variant="subtitle2" color="success.main" gutterBottom>
          長所:
        </Typography>
        <List dense>
          {algorithm.pros.slice(0, 3).map((pro, index) => (
            <ListItem key={index} sx={{ py: 0, pl: 0 }}>
              <ListItemIcon sx={{ minWidth: 20 }}>
                <CheckCircleIcon fontSize="small" color="success" />
              </ListItemIcon>
              <ListItemText primary={pro} />
            </ListItem>
          ))}
        </List>

        <Typography variant="subtitle2" color="info.main" gutterBottom mt={1}>
          適用場面:
        </Typography>
        <List dense>
          {algorithm.best_for.slice(0, 2).map((use, index) => (
            <ListItem key={index} sx={{ py: 0, pl: 0 }}>
              <ListItemIcon sx={{ minWidth: 20 }}>
                <InfoIcon fontSize="small" color="info" />
              </ListItemIcon>
              <ListItemText primary={use} />
            </ListItem>
          ))}
        </List>

        {algorithm.note && (
          <Alert severity="warning" sx={{ mt: 2 }}>
            <Typography variant="caption">{algorithm.note}</Typography>
          </Alert>
        )}
      </CardContent>
    </Card>
  );

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" p={4}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Box>
      {/* ヘッダー */}
      <Box mb={4}>
        <Typography variant="h4" component="h1" gutterBottom>
          🤖 MLアルゴリズムカタログ
        </Typography>
        <Typography variant="body1" color="text.secondary">
          利用可能な機械学習アルゴリズムの詳細情報と特徴を確認できます。
        </Typography>
      </Box>

      {/* 統計情報 */}
      {statistics && (
        <Paper sx={{ p: 3, mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            📊 統計情報
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={3}>
              <Box textAlign="center">
                <Typography variant="h3" color="primary">
                  {statistics.total}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  総アルゴリズム数
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box textAlign="center">
                <Typography variant="h3" color="info.main">
                  {statistics.probabilityCount}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  確率予測対応
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box textAlign="center">
                <Typography variant="h3" color="success.main">
                  {statistics.featureImportanceCount}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  特徴量重要度対応
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box textAlign="center">
                <Typography variant="h3" color="secondary.main">
                  {statistics.byType.length}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  アルゴリズムタイプ
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      )}

      {/* タブ */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)}>
          <Tab label="全アルゴリズム" />
          <Tab label="タイプ別" />
          <Tab label="機能別" />
        </Tabs>
      </Box>

      {/* 全アルゴリズムタブ */}
      <TabPanel value={tabValue} index={0}>
        {/* フィルター */}
        <Box mb={3}>
          <Grid container spacing={2}>
            <Grid item xs={12} md={8}>
              <TextField
                fullWidth
                label="検索"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="アルゴリズム名、説明、特徴で検索..."
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <FormControl fullWidth>
                <InputLabel>タイプ</InputLabel>
                <Select
                  value={selectedType}
                  onChange={(e) => setSelectedType(e.target.value)}
                  label="タイプ"
                >
                  <MenuItem value="all">すべて</MenuItem>
                  {statistics?.byType.map(({ type }) => (
                    <MenuItem key={type} value={type.toLowerCase().replace(' ', '_')}>
                      {type}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </Box>

        {/* アルゴリズム一覧 */}
        <Grid container spacing={3}>
          {filteredAlgorithms.map(algorithm => (
            <Grid item xs={12} md={6} lg={4} key={algorithm.name}>
              <AlgorithmCard algorithm={algorithm} />
            </Grid>
          ))}
        </Grid>

        {filteredAlgorithms.length === 0 && (
          <Alert severity="info" sx={{ mt: 2 }}>
            条件に一致するアルゴリズムが見つかりませんでした。
          </Alert>
        )}
      </TabPanel>

      {/* タイプ別タブ */}
      <TabPanel value={tabValue} index={1}>
        {Object.entries(algorithmsByType).map(([type, algos]) => (
          <Box key={type} mb={4}>
            <Typography variant="h5" gutterBottom>
              {getTypeIcon(algos[0]?.type)} {type} ({algos.length}個)
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <Grid container spacing={2}>
              {algos.map(algorithm => (
                <Grid item xs={12} md={6} lg={4} key={algorithm.name}>
                  <AlgorithmCard algorithm={algorithm} />
                </Grid>
              ))}
            </Grid>
          </Box>
        ))}
      </TabPanel>

      {/* 機能別タブ */}
      <TabPanel value={tabValue} index={2}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom color="info.main">
                  🎯 確率予測対応アルゴリズム
                </Typography>
                <Typography variant="body2" color="text.secondary" mb={2}>
                  予測確率を出力できるアルゴリズム
                </Typography>
                <List>
                  {algorithms.filter(a => a.has_probability_prediction).map(algo => (
                    <ListItem key={algo.name}>
                      <ListItemText 
                        primary={algo.display_name}
                        secondary={algo.description}
                      />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom color="success.main">
                  📊 特徴量重要度対応アルゴリズム
                </Typography>
                <Typography variant="body2" color="text.secondary" mb={2}>
                  特徴量の重要度を算出できるアルゴリズム
                </Typography>
                <List>
                  {algorithms.filter(a => a.has_feature_importance).map(algo => (
                    <ListItem key={algo.name}>
                      <ListItemText 
                        primary={algo.display_name}
                        secondary={algo.description}
                      />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>
    </Box>
  );
};

export default AlgorithmCatalog;
